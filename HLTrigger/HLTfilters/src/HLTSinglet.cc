/** \class HLTSinglet
 *
 * See header file for documentation
 *
 *  $Date: 2011/05/01 14:41:36 $
 *  $Revision: 1.10 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "HLTrigger/HLTfilters/interface/HLTSinglet.h"


// extract the candidate type
template<typename T, int Tid>
trigger::TriggerObjectType getObjectType(const T &) {
  return static_cast<trigger::TriggerObjectType>(Tid);
}

// specialize for type l1extra::L1EmParticle
template<int Tid>
trigger::TriggerObjectType getObjectType(const l1extra::L1EmParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1EmParticle::kIsolated:
      return trigger::TriggerL1IsoEG;
    case l1extra::L1EmParticle::kNonIsolated:
      return trigger::TriggerL1NoIsoEG;
    default:
      return static_cast<trigger::TriggerObjectType>(Tid);
  }
}

// specialize for type l1extra::L1EtMissParticle
template<int Tid>
trigger::TriggerObjectType getObjectType(const l1extra::L1EtMissParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1EtMissParticle::kMET:
      return trigger::TriggerL1ETM;
    case l1extra::L1EtMissParticle::kMHT:
      return trigger::TriggerL1HTM;
    default:
      return static_cast<trigger::TriggerObjectType>(Tid);
  }
}

// specialize for type l1extra::L1JetParticle
template<int Tid>
trigger::TriggerObjectType getObjectType(const l1extra::L1JetParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1JetParticle::kCentral:
      return trigger::TriggerL1CenJet;
    case l1extra::L1JetParticle::kForward:
      return trigger::TriggerL1ForJet;
    case l1extra::L1JetParticle::kTau:
      return trigger::TriggerL1TauJet;
    default:
      return static_cast<trigger::TriggerObjectType>(Tid);
  }
}


//
// constructors and destructor
//
template<typename T, int Tid>
HLTSinglet<T,Tid>::HLTSinglet(const edm::ParameterSet& iConfig) :
  inputTag_ (iConfig.template getParameter<edm::InputTag>("inputTag")),
  saveTags_  (iConfig.template getParameter<bool>("saveTags")),
  min_Pt_   (iConfig.template getParameter<double>       ("MinPt"   )),
  max_Eta_  (iConfig.template getParameter<double>       ("MaxEta"  )),
  min_N_    (iConfig.template getParameter<int>          ("MinN"    ))
{
   LogDebug("") << "Input/ptcut/etacut/ncut : "
		<< inputTag_.encode() << " "
		<< min_Pt_ << " " << max_Eta_ << " " << min_N_ ;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

template<typename T, int Tid>
HLTSinglet<T,Tid>::~HLTSinglet()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T, int Tid> 
bool
HLTSinglet<T,Tid>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
   if (saveTags_) filterobject->addCollectionTag(inputTag_);
   // Ref to Candidate object to be recorded in filter object
   TRef ref;


   // get hold of collection of objects
   Handle<TCollection> objects;
   iEvent.getByLabel (inputTag_,objects);

   // look at all objects, check cuts and add to filter object
   int n(0);
   typename TCollection::const_iterator i ( objects->begin() );
   for (; i!=objects->end(); i++) {
     if ( (i->pt() >= min_Pt_) && 
	  ( (max_Eta_ < 0.0) || (std::abs(i->eta()) <= max_Eta_) ) ) {
       n++;
       ref=TRef(objects,distance(objects->begin(),i));
       filterobject->addObject(getObjectType<T, Tid>(*i),ref);
     }
   }

   // filter decision
   bool accept(n>=min_N_);

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}


