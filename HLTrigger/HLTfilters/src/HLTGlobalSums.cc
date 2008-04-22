/** \class HLT1GlobalSums
 *
 * See header file for documentation
 *
 *  $Date: 2008/01/09 14:16:15 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLTGlobalSums.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<cmath>

//
// constructors and destructor
//
template<typename T, int Tid>
HLTGlobalSums<T,Tid>::HLTGlobalSums(const edm::ParameterSet& iConfig) :
  inputTag_   (iConfig.template getParameter<edm::InputTag>("inputTag")),
  observable_ (iConfig.template getParameter<std::string>("observable")),
  min_        (iConfig.template getParameter<double>("Min")),
  max_        (iConfig.template getParameter<double>("Max")),
  min_N_      (iConfig.template getParameter<int>("MinN"))
{
   LogDebug("") << "InputTags and cuts : " 
		<< inputTag_.encode() << " " << observable_
		<< " Range [" << min_ << " " << max_ << "]"
                << " MinN =" << min_N_
     ;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

template<typename T, int Tid>
HLTGlobalSums<T,Tid>::~HLTGlobalSums()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T, int Tid> 
bool
HLTGlobalSums<T,Tid>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   typedef vector<T> TCollection;
   typedef Ref<TCollection> TRef;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterobject (new TriggerFilterObjectWithRefs(path(),module()));
   filterobject->addCollectionTag(inputTag_);
   // Ref to Candidate object to be recorded in filter object
   TRef ref;


   // get hold of MET product from Event
   Handle<TCollection>   objects;
   iEvent.getByLabel(inputTag_,objects);
   if (!objects.isValid()) {
     LogDebug("") << inputTag_ << " collection not found!";
     iEvent.put(filterobject);
     return false;
   }

   LogDebug("") << "Size of MET collection: " << objects->size();
   if (objects->size()==0) {
     LogDebug("") << "MET collection does not contain a MET object!";
   } else if (objects->size()>1) {
     LogDebug("") << "MET collection contains more than one MET object!";
   }

   int n(0);
   double value(0.0);
   typename TCollection::const_iterator ibegin(objects->begin());
   typename TCollection::const_iterator iend(objects->end());
   typename TCollection::const_iterator iter;
   for (iter=ibegin; iter!=iend; iter++) {

     // get hold of value of observable to cut on
     if (observable_=="sumEt") {
       value=iter->sumEt();
     } else if (observable_=="e_longitudinal") {
       value=iter->e_longitudinal();
     } else if (observable_=="mEtSig") {
       value=iter->mEtSig();
     } else {
       value=0.0;
     }

     value=abs(value);

     if ( ( (min_<0.0) || (min_<=value) ) &&
	  ( (max_<0.0) || (value<=max_) ) ) {
       n++;
       ref=TRef(objects,distance(ibegin,iter));
       filterobject->addObject(Tid,ref);
     }

   }

   // filter decision
   const bool accept(n>=min_N_);

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}
