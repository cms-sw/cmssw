/** \class HLTSinglet
 *
 * See header file for documentation
 *
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

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

// extract the candidate type
template<typename T>
int getObjectType(const T &) {
  return 0;
}

// specialize for type l1extra::L1EmParticle
template<typename T>
int getObjectType(const l1extra::L1EmParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1EmParticle::kIsolated:
      return trigger::TriggerL1IsoEG;
    case l1extra::L1EmParticle::kNonIsolated:
      return trigger::TriggerL1NoIsoEG;
    default:
      return 0;
  }
}

// specialize for type l1extra::L1EtMissParticle
template<typename T>
int getObjectType(const l1extra::L1EtMissParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1EtMissParticle::kMET:
      return trigger::TriggerL1ETM;
    case l1extra::L1EtMissParticle::kMHT:
      return trigger::TriggerL1HTM;
    default:
      return 0;
  }
}

// specialize for type l1extra::L1JetParticle
template<typename T>
int getObjectType(const l1extra::L1JetParticle & candidate) {
  switch (candidate.type()) {
    case l1extra::L1JetParticle::kCentral:
      return trigger::TriggerL1CenJet;
    case l1extra::L1JetParticle::kForward:
      return trigger::TriggerL1ForJet;
    case l1extra::L1JetParticle::kTau:
      return trigger::TriggerL1TauJet;
    default:
      return 0;
  }
}


//
// constructors and destructor
//
template<typename T>
HLTSinglet<T>::HLTSinglet(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputTag_    (iConfig.template getParameter<edm::InputTag>("inputTag")),
  inputToken_  (consumes<std::vector<T> >(inputTag_)),
  triggerType_ (iConfig.template getParameter<int>("triggerType")),
  min_N_    (iConfig.template getParameter<int>          ("MinN"    )),
  min_E_    (iConfig.template getParameter<double>       ("MinE"    )),
  min_Pt_   (iConfig.template getParameter<double>       ("MinPt"   )),
  min_Mass_ (iConfig.template getParameter<double>       ("MinMass" )),
  max_Eta_  (iConfig.template getParameter<double>       ("MaxEta"  ))
{
   LogDebug("") << "Input/ptcut/etacut/ncut : "
		<< inputTag_.encode() << " "
		<< min_E_ << " " << min_Pt_ << " " << min_Mass_ << " "
		<< max_Eta_ << " " << min_N_ ;
}

template<typename T>
HLTSinglet<T>::~HLTSinglet()
{
}

template<typename T>
void
HLTSinglet<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltCollection"));
  desc.add<int>("triggerType",0);
  desc.add<double>("MinE",-1.0);
  desc.add<double>("MinPt",-1.0);
  desc.add<double>("MinMass",-1.0);
  desc.add<double>("MaxEta",-1.0);
  desc.add<int>("MinN",1);
  descriptions.add(defaultModuleLabel<HLTSinglet<T>>(), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTSinglet<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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
   iEvent.getByToken(inputToken_,objects);

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
       int tid = getObjectType<T>(*i);
       if (tid == 0)
         tid = triggerType_;
       filterproduct.addObject(tid, ref);
     }
   }

   // filter decision
   bool accept(n>=min_N_);

   return accept;
}
