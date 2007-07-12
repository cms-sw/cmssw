/** \class HLTHighLevel
 *
 * See header file for documentation
 *
 *  $Date: 2007/06/19 12:31:19 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

//
// constructors and destructor
//
HLTHighLevel::HLTHighLevel(const edm::ParameterSet& iConfig) :
  inputTag_ (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  triggerNames_(),
  andOr_    (iConfig.getParameter<bool> ("andOr" )),
  n_        (0)

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

// ------------ method called to produce the data  ------------
bool
HLTHighLevel::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   const string invalid("@@invalid@@");

   // get hold of TriggerResults Object
   Handle<TriggerResults> trh;
   try {iEvent.getByLabel(inputTag_,trh);} catch(...) {;}
   if (trh.isValid()) {
     LogDebug("") << "TriggerResults found, number of HLT paths: " << trh->size();
   } else {
     LogDebug("") << "TriggerResults product not found - returning result=false!";
     return false;
   }

   // get hold of trigger names - based on TriggerResults object!
   triggerNames_.init(*trh);

   unsigned int n(n_);
   for (unsigned int i=0; i!=n; i++) {
     HLTPathsByIndex_[i]=triggerNames_.triggerIndex(HLTPathsByName_[i]);
   }
   
   // for empty input vectors (n==0), default to all HLT trigger paths!
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
   LogDebug("") << "HLT trigger paths: " + inputTag_.encode()
		<< " - Number requested: " << n
		<< " - andOr mode: " << andOr_;
   if (n>0) {
     LogDebug("") << "  HLT trigger paths requested: index, name and valididty:";
     for (unsigned int i=0; i!=n; i++) {
       LogTrace("") << " " << HLTPathsByIndex_[i]
		    << " " << HLTPathsByName_[i]
		    << " " << ( (HLTPathsByIndex_[i]<trh->size()) && (HLTPathsByName_[i]!=invalid) );
     }
   }

   // count number of requested HLT paths which have fired
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) {
     if (HLTPathsByIndex_[i]<trh->size()) {
       if (trh->accept(HLTPathsByIndex_[i])) {
	 fired++;
       }
     }
   }

   // Boolean filter result
   const bool accept( ((!andOr_) && (fired==n)) ||
		      (( andOr_) && (fired!=0)) );
   LogDebug("") << "Accept = " << accept;

   return accept;

}
