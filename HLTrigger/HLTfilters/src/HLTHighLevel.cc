/** \class HLTHighLevel
 *
 * See header file for documentation
 *
 *  $Date: 2008/10/17 09:57:43 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

// static functions (partly taken from EventSelector) - see below
bool is_glob(const std::string & pattern);
  
std::string glob2reg(const std::string & pattern);

std::vector< std::vector<std::string>::const_iterator > 
matching_triggers(const std::vector<std::string> & triggers, const std::string & pattern);

//
// constructors and destructor
//
HLTHighLevel::HLTHighLevel(const edm::ParameterSet& iConfig) :
  inputTag_     (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  triggerNames_ (),
  andOr_        (iConfig.getParameter<bool> ("andOr")),
  throw_        (iConfig.getUntrackedParameter<bool> ("throw", true)),
  HLTPatterns_  (iConfig.getParameter<std::vector<std::string> >("HLTPaths")),
  HLTPathsByName_(),
  HLTPathsByIndex_()
{
  // names and slot numbers are computed during the event loop, 
  // as they need to access the TriggerNames object via the TriggerResults
}

HLTHighLevel::~HLTHighLevel()
{
}

//
// member functions
//

// Initialize the internal trigger path representation (names and indices) from the 
// patterns specified in the configuration.
// This needs to be called once at startup and whenever the trigger table has changed
void HLTHighLevel::init(const edm::TriggerResults & result)
{
   unsigned int n;
   
   // clean up old data
   HLTPathsByName_.clear();
   HLTPathsByIndex_.clear();

   if (HLTPatterns_.empty()) {
     // for empty input vector, default to all HLT trigger paths
     n = result.size();
     HLTPathsByName_.resize(n);
     HLTPathsByIndex_.resize(n);
     for (unsigned int i = 0; i < n; ++i) {
       HLTPathsByName_[i] = triggerNames_.triggerName(i);
       HLTPathsByIndex_[i] = i;
     }
   } else {
     // otherwise, expand wildcards in trigger names...
     BOOST_FOREACH(const std::string & pattern, HLTPatterns_) {
       if (is_glob(pattern)) {
         // found a glob pattern, expand it
         std::vector< std::vector<std::string>::const_iterator > matches = matching_triggers(triggerNames_.triggerNames(), glob2reg(pattern));
         if (matches.empty()) {
           // pattern does not match any trigger paths
           LogDebug("") << "requested pattern does not match any HLT paths: " << pattern;
         } else {
           // store the matching patterns
           BOOST_FOREACH(std::vector<std::string>::const_iterator match, matches)
             HLTPathsByName_.push_back(*match);
         }
       } else {
         // found a trigger name, just copy it
         HLTPathsByName_.push_back(pattern);
       }
     }
     n = HLTPathsByName_.size();
   
     // ...and get hold of trigger indices
     HLTPathsByIndex_.resize(n);
     for (unsigned int i = 0; i < HLTPathsByName_.size(); i++) {
       HLTPathsByIndex_[i] = triggerNames_.triggerIndex(HLTPathsByName_[i]);
       if (HLTPathsByIndex_[i] >= result.size()) {
         // trigger path not found
         LogDebug("") << "requested HLT path does not exist: " << HLTPathsByName_[i];
         HLTPathsByIndex_[i] = (unsigned int) -1;
       }
     }
   }
   
   // report on what is finally used
   LogDebug("") << "HLT trigger paths: " + inputTag_.encode()
                << " - Number of paths: " << n
                << " - andOr mode: " << andOr_
                << " - throw mode: " << throw_;
   
   LogTrace("") << "The HLT trigger paths (# index name):";
   for (unsigned int i = 0; i < n; ++i)
     if (HLTPathsByIndex_[i] == (unsigned int) -1)
       LogTrace("") << "    n/a   " << HLTPathsByName_[i];
     else
       LogTrace("") << "    " << std::setw(4) << HLTPathsByIndex_[i] << "  " << HLTPathsByName_[i];

}

// ------------ method called to produce the data  ------------
bool
HLTHighLevel::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;

   // get hold of TriggerResults Object
   Handle<TriggerResults> trh;
   iEvent.getByLabel(inputTag_, trh);
   if (trh.isValid()) {
     LogDebug("") << "TriggerResults found, number of HLT paths: " << trh->size();
   } else {
     LogDebug("") << "TriggerResults product not found - returning result=false!";
     return false;
   }

   // init the TriggerNames with the TriggerResults
   bool config_changed = triggerNames_.init(*trh);

   // if this is the first event or the HLT table has changed (re)run th initialization stuff
   if (config_changed)
     init(*trh);  

   unsigned int n     = HLTPathsByName_.size();
   unsigned int nbad  = 0;
   unsigned int fired = 0;

   // count invalid and fired triggers
   for (unsigned int i = 0; i < n; i++)
     if (HLTPathsByIndex_[i] == (unsigned int) -1)
       ++nbad;
     else if (trh->accept(HLTPathsByIndex_[i]))
       ++fired;

   if ((nbad > 0) and (config_changed or throw_)) {
     // only generate the error message if it's actually going to be used
     std::string message;

     for (unsigned int i = 0; i < n; i++)
       if (HLTPathsByIndex_[i] == (unsigned int) -1)
         message += HLTPathsByName_[i] + " ";

     if (config_changed) {
       LogTrace("")
         << " HLTHighLevel [instance: " << *moduleLabel()
         << " - path: " << *pathName()
         << "] configured with " << nbad
         << "/" << n
         << " unknown HLT path names: " << message;
     }

     if (throw_) {
       throw cms::Exception("Configuration")
         << " HLTHighLevel [instance: " << *moduleLabel()
         << " - path: " << *pathName()
         << "] configured with " << nbad
         << "/" << n
         << " unknown HLT path names: " << message;
     }
   }

   // Boolean filter result (always at least one trigger)
   const bool accept( (fired > 0) and ( andOr_ or (fired == n-nbad) ) );
   LogDebug("") << "Accept = " << std::boolalpha << accept;

   return accept;
}

// ----------------------------------------------------------------------------
// static functions copied from EventSelector

bool is_glob(const std::string & pattern)
{
  return (pattern.find_first_of("*?") != pattern.npos);
}

std::string glob2reg(const std::string & pattern) 
{
  std::string regexp = pattern;
  boost::replace_all(regexp, "*", ".*");
  boost::replace_all(regexp, "?", ".");
  return regexp;
}

std::vector< std::vector<std::string>::const_iterator > 
matching_triggers(const std::vector<std::string> & triggers, const std::string & pattern) 
{
  std::vector< std::vector<std::string>::const_iterator > matches;
  boost::regex regexp( glob2reg(pattern) );
  for (std::vector<std::string>::const_iterator i = triggers.begin(); i != triggers.end(); ++i)
    if (boost::regex_match((*i), regexp)) 
      matches.push_back(i);

  return matches;
}
