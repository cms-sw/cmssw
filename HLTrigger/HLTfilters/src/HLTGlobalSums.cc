/** \class HLT1GlobalSums
 *
 * See header file for documentation
 *
 *  $Date: 2012/02/24 13:13:47 $
 *  $Revision: 1.15 $
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
template<typename T>
HLTGlobalSums<T>::HLTGlobalSums(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputTag_   (iConfig.template getParameter<edm::InputTag>("inputTag")),
  triggerType_(iConfig.template getParameter<int>("triggerType")),
  observable_ (iConfig.template getParameter<std::string>("observable")),
  min_        (iConfig.template getParameter<double>("Min")),
  max_        (iConfig.template getParameter<double>("Max")),
  min_N_      (iConfig.template getParameter<int>("MinN")),
  tid_(triggerType_)
{
   LogDebug("") << "InputTags and cuts : " 
		<< inputTag_.encode() << " "
		<< triggerType_ << " "
		<< observable_
		<< " Range [" << min_ << " " << max_ << "]"
                << " MinN =" << min_N_ ;

   if (observable_=="sumEt") {
     tid_=triggerType_;
   } else if (observable_=="mEtSig") {
     if (triggerType_==trigger::TriggerTET) {
       tid_=trigger::TriggerMETSig;
     } else if (triggerType_==trigger::TriggerTHT) {
       tid_=trigger::TriggerMHTSig;
     } else {
       tid_=triggerType_;
     }
   } else if (observable_=="e_longitudinal") {
     if (triggerType_==trigger::TriggerTET) {
       tid_=trigger::TriggerELongit;
     } else if (triggerType_==trigger::TriggerTHT) {
       tid_=trigger::TriggerHLongit;
     } else {
       tid_=triggerType_;
     }
   } else {
     tid_=triggerType_;
   }
}

template<typename T>
HLTGlobalSums<T>::~HLTGlobalSums()
{
}

template<typename T>
void
HLTGlobalSums<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltCollection"));
  desc.add<int>("triggerType",0);
  desc.add<std::string>("observable","");
  desc.add<double>("Min",-1e125);
  desc.add<double>("Max",+1e125);
  desc.add<int>("MinN",1);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTGlobalSums<T>).name()),desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T> 
bool
HLTGlobalSums<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
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
   if (saveTags()) filterproduct.addCollectionTag(inputTag_);
   // Ref to Candidate object to be recorded in filter object
   TRef ref;


   // get hold of MET product from Event
   Handle<TCollection>   objects;
   iEvent.getByLabel(inputTag_,objects);
   if (!objects.isValid()) {
     LogDebug("") << inputTag_ << " collection not found!";
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
     if ( (tid_==TriggerTET) || (tid_==TriggerTHT) ) {
       value=iter->sumEt();
     } else if ( (tid_==TriggerMETSig) || (tid_==TriggerMHTSig) ) {
       value=iter->mEtSig();
     } else if ( (tid_==TriggerELongit) || (tid_==TriggerHLongit) ) {
       value=iter->e_longitudinal();
     } else {
       value=0.0;
     }

     value=std::abs(value);

     if ( ( (min_<0.0) || (min_<=value) ) &&
	  ( (max_<0.0) || (value<=max_) ) ) {
       n++;
       ref=TRef(objects,distance(ibegin,iter));
       filterproduct.addObject(tid_,ref);
     }

   }

   // filter decision
   const bool accept(n>=min_N_);

   return accept;
}
