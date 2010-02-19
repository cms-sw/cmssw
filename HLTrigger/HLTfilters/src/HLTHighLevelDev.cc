/** \class HLTHighLevelDev
 *
 * See header file for documentation
 *
 *  $Date: 2009/11/13 19:32:23 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <boost/foreach.hpp>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// needed for trigger bits from EventSetup as in ALCARECO paths
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


#include "HLTrigger/HLTfilters/interface/HLTHighLevelDev.h"

//
// constructors and destructor
//
HLTHighLevelDev::HLTHighLevelDev(const edm::ParameterSet& iConfig) :
  inputTag_     (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  triggerNames_ (),
  andOr_        (iConfig.getParameter<bool> ("andOr")),
  throw_        (iConfig.getParameter<bool> ("throw")),
  eventSetupPathsKey_(iConfig.getParameter<std::string>("eventSetupPathsKey")),
  watchAlCaRecoTriggerBitsRcd_(0),
  HLTPatterns_  (iConfig.getParameter<std::vector<std::string> >("HLTPaths")),
  HLTPrescales_ (iConfig.getParameter<std::vector<uint32_t> >("HLTPathsPrescales")),
  HLTPrescalesExpanded_(), 
  HLTOverallPrescale_ (iConfig.getParameter<uint32_t>("HLTOverallPrescale")),
  HLTPathsByName_(),
  HLTPathsByIndex_()
{
  // names and slot numbers are computed during the event loop, 
  // as they need to access the TriggerNames object via the TriggerResults

  if (eventSetupPathsKey_.size()) {
    // If paths come from eventsetup, we must watch for IOV changes.
    if (!HLTPatterns_.empty()) {
      // We do not want double trigger path setting, so throw!
      throw cms::Exception("Configuration")
        << " HLTHighLevelDev instance: "<< iConfig.getParameter<std::string>("@module_label")
        << "\n configured with " << HLTPatterns_.size() << " HLTPaths and\n"
        << " eventSetupPathsKey " << eventSetupPathsKey_ << ", choose either of them.";
    }
    watchAlCaRecoTriggerBitsRcd_ = new edm::ESWatcher<AlCaRecoTriggerBitsRcd>;
  }
}

HLTHighLevelDev::~HLTHighLevelDev()
{
  delete watchAlCaRecoTriggerBitsRcd_; // safe on null pointer...
}

//
// member functions
//

// Initialize the internal trigger path representation (names and indices) from the 
// patterns specified in the configuration.
// This needs to be called once at startup, whenever the trigger table has changed
// or in case of paths from eventsetup and IOV changed
void HLTHighLevelDev::init(const edm::TriggerResults & result, const edm::EventSetup& iSetup)
{
  unsigned int n;

  // clean up old data
  HLTPathsByName_.clear();
  HLTPathsByIndex_.clear();

  // Overwrite paths from EventSetup via AlCaRecoTriggerBitsRcd if configured:
  if (eventSetupPathsKey_.size()) {
    HLTPatterns_ = this->pathsFromSetup(eventSetupPathsKey_, iSetup);
  }

  // check the HLTPatterns_ and HLTPrescales_ have the same number of elements
  if (HLTPatterns_.size() != HLTPrescales_.size())
    throw cms::Exception("Configuration") << "HLTPrescales must have the same number of entries as HLTPatterns.";

  if (HLTPatterns_.empty()) {
    // for empty input vector, default to all HLT trigger paths
    n = result.size();
    HLTPathsByName_.resize(n);
    HLTPathsByIndex_.resize(n);
    HLTPrescalesExpanded_.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      HLTPathsByName_[i] = triggerNames_.triggerName(i);
      HLTPathsByIndex_[i] = i;
      HLTPrescalesExpanded_[i] = 1;
    }
  } else {
    // otherwise, expand wildcards in trigger names...
    std::vector<uint32_t>::const_iterator i_scale = HLTPrescales_.begin();
    BOOST_FOREACH(const std::string & pattern, HLTPatterns_) {
      uint32_t scale = *(i_scale++);
      if (edm::is_glob(pattern)) {
        // found a glob pattern, expand it
        std::vector< std::vector<std::string>::const_iterator > matches = edm::regexMatch(triggerNames_.triggerNames(), pattern);
        if (matches.empty()) {
          // pattern does not match any trigger paths
          if (throw_)
            throw cms::Exception("Configuration") << "requested pattern \"" << pattern <<  "\" does not match any HLT paths";
          else
            edm::LogInfo("Configuration") << "requested pattern \"" << pattern <<  "\" does not match any HLT paths";
        } else {
          // store the matching patterns
          BOOST_FOREACH(std::vector<std::string>::const_iterator match, matches) {
            HLTPathsByName_.push_back(*match);
            HLTPrescalesExpanded_.push_back(scale);
          }
        }
      } else {
        // found a trigger name, just copy it
        HLTPathsByName_.push_back(pattern);
        HLTPrescalesExpanded_.push_back(scale);
      }
    }
    n = HLTPathsByName_.size();

    // resizing and initializing the scalers
    HLTPrescalesScalers.resize(n, 0);
    HLTOverallPrescalesScaler_ = 0;

    // ...and get hold of trigger indices
    bool valid = false;
    HLTPathsByIndex_.resize(n);
    for (unsigned int i = 0; i < HLTPathsByName_.size(); i++) {
      HLTPathsByIndex_[i] = triggerNames_.triggerIndex(HLTPathsByName_[i]);
      if (HLTPathsByIndex_[i] < result.size()) {
        valid = true;
      } else {
        // trigger path not found
        HLTPathsByIndex_[i] = (unsigned int) -1;
        if (throw_)
          throw cms::Exception("Configuration") << "requested HLT path \"" << HLTPathsByName_[i] << "\" does not exist";
        else
          edm::LogInfo("Configuration") << "requested HLT path \"" << HLTPathsByName_[i] << "\" does not exist";
      }
    }

    if (not valid) {
      // no point in throwing - if requested, it should have already happened
      edm::LogWarning("Configuration") << "none of the requested paths and pattern match any HLT path - no events will be selected";
    }
    
  }

  // report on what is finally used
  LogDebug("HLTHighLevelDev") << "HLT trigger paths: " + inputTag_.encode()
    << " - Number of paths: " << n
    << " - andOr mode: " << andOr_
    << " - throw mode: " << throw_;

  LogTrace("HLTHighLevelDev") << "The HLT trigger paths (# index name):";
  for (unsigned int i = 0; i < n; ++i)
    if (HLTPathsByIndex_[i] == (unsigned int) -1)
      LogTrace("HLTHighLevelDev") << "    n/a   " << HLTPathsByName_[i];
    else
      LogTrace("HLTHighLevelDev") << "    " << std::setw(4) << HLTPathsByIndex_[i] << "  " << HLTPathsByName_[i];

}

// ------------ getting paths from EventSetup  ------------
std::vector<std::string>
HLTHighLevelDev::pathsFromSetup(const std::string &key, const edm::EventSetup &iSetup) const
{
  // Get map of strings to concatenated list of names of HLT paths from EventSetup:
  edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
  iSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap &triggerMap = triggerBits->m_alcarecoToTrig;

  TriggerMap::const_iterator listIter = triggerMap.find(key);
  if (listIter == triggerMap.end()) {
    throw cms::Exception("Configuration")
      << " HLTHighLevelDev [instance: " << *moduleLabel() << " - path: " << *pathName()
      << "]: No triggerList with key " << key << " in AlCaRecoTriggerBitsRcd";
  }

  // We must avoid a map<string,vector<string> > in DB for performance reason,
  // so the paths are mapped into one string that we have to decompose:
  return triggerBits->decompose(listIter->second);
}

// ------------ method called to produce the data  ------------
bool
HLTHighLevelDev::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  // get hold of TriggerResults Object
  Handle<TriggerResults> trh;
  iEvent.getByLabel(inputTag_, trh);
  if (trh.isValid()) {
    LogDebug("HLTHighLevelDev") << "TriggerResults found, number of HLT paths: " << trh->size();
  } else {
    LogDebug("HLTHighLevelDev") << "TriggerResults product not found - returning result=false!";
    return false;
  }

  // init the TriggerNames with the TriggerResults
  bool config_changed = triggerNames_.init(*trh);

  // (re)run the initialization stuff if 
  // - this is the first event 
  // - or the HLT table has changed 
  // - or selected trigger bits come from AlCaRecoTriggerBitsRcd and these changed
  if (config_changed or (watchAlCaRecoTriggerBitsRcd_ and watchAlCaRecoTriggerBitsRcd_->check(iSetup))) {
    this->init(*trh, iSetup);
    for (unsigned int i = 0; i < HLTPrescalesScalers.size(); ++i)
      HLTPrescalesScalers[i] = iEvent.id().event() + 1;
    HLTOverallPrescalesScaler_ = iEvent.id().event() + 1;
  }
  unsigned int n     = HLTPathsByName_.size();
  unsigned int nbad  = 0;
  unsigned int fired = 0;
  bool allFired = true;

  // count invalid and fired triggers
  for (unsigned int i = 0; i < n; i++) {
    if (HLTPathsByIndex_[i] == (unsigned int) -1)
      ++nbad;
    else if (trh->accept(HLTPathsByIndex_[i]) and HLTPrescalesExpanded_[i] and (HLTPrescalesScalers[i]++ % HLTPrescalesExpanded_[i] == 0))
      ++fired;
    else
      allFired = false;
  }

  if ((nbad > 0) and (config_changed or throw_)) {
    // only generate the error message if it's actually going to be used
    std::string message;

    for (unsigned int i = 0; i < n; i++)
      if (HLTPathsByIndex_[i] == (unsigned int) -1)
        message += HLTPathsByName_[i] + " ";

    if (config_changed) {
      LogTrace("HLTHighLevelDev")
        << " HLTHighLevelDev [instance: " << *moduleLabel()
        << " - path: " << *pathName()
        << "] configured with " << nbad
        << "/" << n
        << " unknown HLT path names: " << message;
    }

    if (throw_) {
      throw cms::Exception("Configuration")
        << " HLTHighLevelDev [instance: " << *moduleLabel()
        << " - path: " << *pathName()
        << "] configured with " << nbad
        << "/" << n
        << " unknown HLT path names: " << message;
    }
  }

  // Boolean filter result (always at least one trigger)
  // const bool accept( (fired > 0) and ( andOr_ or (fired == n-nbad) ) );
  const bool accept( (fired > 0) and ( andOr_ or allFired ) );
  LogDebug("HLTHighLevelDev") << "Accept = " << std::boolalpha << accept;

  return (accept and HLTOverallPrescale_ and ((HLTOverallPrescalesScaler_++) % HLTOverallPrescale_ == 0));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHighLevelDev);
