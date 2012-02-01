/** \class HLTSinglet
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 14:56:59 $
 *  $Revision: 1.12 $
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

#include <typeinfo>

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
HLTSinglet<T,Tid>::HLTSinglet(const edm::ParameterSet& iConfig) : HLTFilter(iConfig), 
  inputTag_ (iConfig.template getParameter<edm::InputTag>("inputTag")),
  min_E_    (iConfig.template getParameter<double>       ("MinE"    )),
  min_Pt_   (iConfig.template getParameter<double>       ("MinPt"   )),
  min_Mass_ (iConfig.template getParameter<double>       ("MinMass" )),
  max_Eta_  (iConfig.template getParameter<double>       ("MaxEta"  )),
  min_N_    (iConfig.template getParameter<int>          ("MinN"    ))
{
   LogDebug("") << "Input/ptcut/etacut/ncut : "
		<< inputTag_.encode() << " "
		<< min_E_ << " " << min_Pt_ << " " << min_Mass_ << " " 
		<< max_Eta_ << " " << min_N_ ;
}

template<typename T, int Tid>
HLTSinglet<T,Tid>::~HLTSinglet()
{
}

template<typename T, int Tid>
void
HLTSinglet<T,Tid>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltCollection"));
  desc.add<double>("MinE",-1.0);
  desc.add<double>("MinPt",-1.0);
  desc.add<double>("MinMass",-1.0);
  desc.add<double>("MaxEta",-1.0);
  desc.add<int>("MinN",1);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTSinglet<T,Tid>).name()),desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T, int Tid> 
bool
HLTSinglet<T,Tid>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
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


   // get hold of collection of objects
   Handle<TCollection> objects;
   iEvent.getByLabel (inputTag_,objects);

   // look at all objects, check cuts and add to filter object
   int n(0);
   typename TCollection::const_iterator i ( objects->begin() );
   for (; i!=objects->end(); i++) {
     if ( (i->energy() >= min_E_) &&
	  (i->pt() >= min_Pt_) && 
	  (i->mass() >= min_Mass_) && 
	  ( (max_Eta_ < 0.0) || (std::abs(i->eta()) <= max_Eta_) ) ) {
       n++;
       ref=TRef(objects,distance(objects->begin(),i));
       filterproduct.addObject(getObjectType<T, Tid>(*i),ref);
     }
   }

   // filter decision
   bool accept(n>=min_N_);

   return accept;
}
