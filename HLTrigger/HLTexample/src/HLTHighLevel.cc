/** \class HLTHighLevel
 *
 * See header file for documentation
 *
 *  $Date: 2006/09/20 09:46:38 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTHighLevel.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

//
// constructors and destructor
//
HLTHighLevel::HLTHighLevel(const edm::ParameterSet& iConfig) :
  TriggerResultsTag_(iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  andOr_     (iConfig.getParameter<bool> ("andOr" )),
  byName_    (iConfig.getParameter<bool> ("byName")),
  n_         (0)
{
  if (byName_) {
    // get names, then derive slot numbers
    HLTPathByName_= iConfig.getParameter<std::vector<std::string > >("HLTPaths");
    n_=HLTPathByName_.size();
    HLTPathByIndex_.resize(n_);
  } else {
    // get slot numbers, then derive names
    HLTPathByIndex_= iConfig.getParameter<std::vector<unsigned int> >("HLTPaths");
    n_=HLTPathByIndex_.size();
    HLTPathByName_.resize(n_);
  }

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
   try {iEvent.getByLabel(TriggerResultsTag_,trh);} catch(...) {;}
   if (trh.isValid()) {
     LogDebug("") << "TriggerResults found, number of HLT paths: " << trh->size();
   } else {
     LogDebug("") << "TriggerResults product not found - returning result=false!";
     return false;
   }

   // use event data to get the current HLT trigger table
   // this is an ugly hack, the (possibly changing) HLT trigger table 
   // should rather be taken from some runBlock or lumiBlock on file

   unsigned int n(n_);
   if (byName_) {
     for (unsigned int i=0; i!=n; i++) {
       HLTPathByIndex_[i]=trh->find(HLTPathByName_[i]);
     }
   } else {
     for (unsigned int i=0; i!=n; i++) {
       if (HLTPathByIndex_[i]<trh->size()) {
	 HLTPathByName_[i]=trh->name(HLTPathByIndex_[i]);
       } else {
	 HLTPathByName_[i]=invalid;
       }
     }
   }
   
   // for empty input vectors (n==0), default to all HLT trigger paths!
   if (n==0) {
     n=trh->size();
     HLTPathByName_.resize(n);
     HLTPathByIndex_.resize(n);
     for (unsigned int i=0; i!=n; i++) {
       HLTPathByName_[i]=trh->name(i);
       HLTPathByIndex_[i]=i;
     }
   }

   // report on what is finally used
   LogDebug("") << "HLT trigger paths: " +TriggerResultsTag_.encode()
		<< " - Number requested: " << n
		<< " - andOr mode: " << andOr_
		<< " - byName: " << byName_;
   if (n>0) {
     LogDebug("") << "  HLT trigger paths requested: index, name and valididty:";
     for (unsigned int i=0; i!=n; i++) {
       LogTrace("") << " " << HLTPathByIndex_[i]
		    << " " << HLTPathByName_[i]
		    << " " << ( (HLTPathByIndex_[i]<trh->size()) && (HLTPathByName_[i]!=invalid) );
     }
   }

   // count number of requested HLT paths which have fired
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) {
     if (HLTPathByIndex_[i]<trh->size()) {
       if (trh->accept(HLTPathByIndex_[i])) {
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
