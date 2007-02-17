/** \class HLT1GlobalSums
 *
 * See header file for documentation
 *
 *  $Date: 2006/10/04 16:02:42 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTexample/interface/HLTGlobalSums.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "DataFormats/METReco/interface/CaloMET.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<cmath>

//
// constructors and destructor
//
HLTGlobalSums::HLTGlobalSums(const edm::ParameterSet& iConfig) :
  inputTag_   (iConfig.getParameter<edm::InputTag>("inputTag")),
  observable_ (iConfig.getParameter<std::string>("observable")),
  min_        (iConfig.getParameter<double>("Min")),
  max_        (iConfig.getParameter<double>("Max")),
  min_N_      (iConfig.getParameter<int>("MinN"))
{
   LogDebug("") << "InputTags and cuts : " 
		<< inputTag_.encode() << " " << observable_
		<< " Range [" << min_ << " " << max_ << "]"
                << " MinN =" << min_N_
     ;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTGlobalSums::~HLTGlobalSums()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTGlobalSums::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs>
     filterobject (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RefToBase<Candidate> ref;


   // get hold of MET product from Event
   Handle<CaloMETCollection>   mets;
   iEvent.getByLabel(inputTag_,mets);
   if (!mets.isValid()) {
     LogDebug("") << "MET collection not found!";
     iEvent.put(filterobject);
     return false;
   }

   LogDebug("") << "Size of MET collection: " << mets->size();
   if (mets->size()==0) {
     LogDebug("") << "MET collection does not contain a MET object!";
   } else if (mets->size()>1) {
     LogDebug("") << "MET collection contains more than one MET object!";
   }

   int n(0);
   double value(0.0);
   CaloMETCollection::const_iterator amets(mets->begin());
   CaloMETCollection::const_iterator omets(mets->end());
   CaloMETCollection::const_iterator imets;
   for (imets=amets; imets!=omets; imets++) {

     // get hold of value of observable to cut on
     if (observable_=="sumEt") {
       value=imets->sumEt();
     } else if (observable_=="e_longitudinal") {
       value=imets->e_longitudinal();
     } else if (observable_=="mEtSig") {
       value=imets->mEtSig();
     } else {
       value=0.0;
     }

     value=abs(value);

     if ( ( (min_<0.0) || (min_<=value) ) &&
	  ( (max_<0.0) || (value<=max_) ) ) {
       n++;
       ref=RefToBase<Candidate>(CaloMETRef(mets,distance(amets,imets)));
       filterobject->putParticle(ref);
     }

   }

   // filter decision
   const bool accept(n>=min_N_);

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}
