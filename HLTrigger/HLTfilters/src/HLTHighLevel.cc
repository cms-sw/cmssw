/** \class HLTHighLevel
 *
 * See header file for documentation
 *
 *  $Date: 2008/01/09 14:30:05 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

//
// constructors and destructor
//
HLTHighLevel::HLTHighLevel(const edm::ParameterSet& iConfig) :
  inputTag_ (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  triggerNames_(),
  andOr_    (iConfig.getParameter<bool> ("andOr" )),
  throw_    (iConfig.getUntrackedParameter<bool> ("throw",true)),
  n_        (0),
  first_    (true)

{
  // get names from module parameters, then derive slot numbers
  HLTPathsByName_= iConfig.getParameter<std::vector<std::string > >("HLTPaths");
  n_=HLTPathsByName_.size();
  HLTPathsByIndex_.resize(n_);

  // this is a user/analysis filter: it places no product into the event!

}

HLTHighLevel::~HLTHighLevel()
{
}

//
// member functions
//
bool
HLTHighLevel:: beginRun(edm::Run& iRun, const edm::EventSetup& iSetup)
{
  first_=true;
  return true;
}

// ------------ method called to produce the data  ------------
bool
HLTHighLevel::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;

   const string invalid("@@invalid@@");

   // get hold of TriggerResults Object
   Handle<TriggerResults> trh;
   iEvent.getByLabel(inputTag_,trh);
   if (trh.isValid()) {
     LogDebug("") << "TriggerResults found, number of HLT paths: " << trh->size();
   } else {
     LogDebug("") << "TriggerResults product not found - returning result=false!";
     return false;
   }

   // get hold of trigger names and indices - based on TriggerResults object!
   triggerNames_.init(*trh);
   unsigned int n(n_);
   for (unsigned int i=0; i!=n; i++) {
     HLTPathsByIndex_[i]=triggerNames_.triggerIndex(HLTPathsByName_[i]);
   }
   
   // for empty input vector (n==0), default to all HLT trigger paths!
   if (n==0) {
     n=trh->size();
     HLTPathsByName_.resize(n);
     HLTPathsByIndex_.resize(n);
     for (unsigned int i=0; i!=n; i++) {
       HLTPathsByName_[i]=triggerNames_.triggerName(i);
       HLTPathsByIndex_[i]=i;
     }
   }

   // report on what is finally used
   if (first_) {
     LogDebug("") << "HLT trigger paths: " + inputTag_.encode()
		  << " - Number of paths: " << n
		  << " - andOr mode: " << andOr_
		  << " - throw mode: " << throw_;
     LogDebug("") << "   The HLT trigger paths (# index name):";
   }

   unsigned int nbad(0);
   string message("   ");
   for (unsigned int i=0; i!=n; i++) {
     if (first_) {
       LogTrace("") << " " << i 
		    << " " << HLTPathsByIndex_[i]
		    << " " << HLTPathsByName_[i];
     }
     if (HLTPathsByIndex_[i]>=trh->size()) {
       nbad++;
       message=message+" "+HLTPathsByName_[i];
     }
   }

   if (nbad>0) {
     if (first_) {
       LogTrace("") << "  Unknown Triggers: " << message;
       cout
	 << " HLTHighLevel [instance: " << *moduleLabel()
	 << " - path: " << *pathName()
	 << "] configured with " << nbad
	 << "/" << n
	 << " unknown HLT path names: " << message
	 << "\n";
       first_=false;
     }
     if (throw_) {
       throw cms::Exception("Configuration")
	 << " HLTHighLevel [instance: " << *moduleLabel()
	 << " - path: " << *pathName()
	 << "] configured with " << nbad
	 << "/" << n
	 << " unknown HLT path names: " << message
	 << "\n";
     }
   }

   // count number of requested known HLT paths which have fired
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) {
     if (HLTPathsByIndex_[i]<trh->size()) {
       if (trh->accept(HLTPathsByIndex_[i])) {
	 fired++;
       }
     }
   }

   // Boolean filter result (always at least one trigger)
   const bool accept( (fired>0) && ( andOr_ || (fired==n-nbad) ) );
   LogDebug("") << "Accept = " << accept;

   return accept;

}
