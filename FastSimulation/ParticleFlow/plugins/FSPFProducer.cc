#include "FastSimulation/ParticleFlow/plugins/FSPFProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;
using namespace reco;


FSPFProducer::FSPFProducer(const edm::ParameterSet& iConfig) {
 
  labelPFCandidateCollection_ = iConfig.getParameter < edm::InputTag > ("pfCandidates");
  
  barrel_correction = iConfig.getParameter<double>("barrel_correction");
  endcap_correction = iConfig.getParameter<double>("endcap_correction");
 
  // register products
  produces<reco::PFCandidateCollection>();
  
}

FSPFProducer::~FSPFProducer() {}

void 
FSPFProducer::beginJob() {}

void 
FSPFProducer::beginRun(edm::Run & run,
		     const edm::EventSetup & es) {}

void 
FSPFProducer::produce(Event& iEvent,
		    const EventSetup& iSetup) {
  
  Handle < reco::PFCandidateCollection > pfCandidates;
  iEvent.getByLabel (labelPFCandidateCollection_, pfCandidates);

  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection(new PFCandidateCollection);   

  LogDebug("FSPFProducer")<<"START event: "
			<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;   

  double theNeutralFraction, px, py, pz, energy;
  reco::PFCandidateCollection::const_iterator  itCand =  pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator  itCandEnd = pfCandidates->end();
  for( ; itCand != itCandEnd; itCand++) {
    pOutputCandidateCollection->push_back(*itCand);
    if(itCand->particleId() == reco::PFCandidate::h){
      theNeutralFraction = (itCand->eta()>-1.5 && itCand->eta()<=1.5)? barrel_correction : endcap_correction;
      px = theNeutralFraction*itCand->px();
      py = theNeutralFraction*itCand->py();
      pz = theNeutralFraction*itCand->pz();
      energy = sqrt(px*px + py*py + pz*pz); 
      math::XYZTLorentzVector momentum(px,py,pz,energy);
      // create a PFCandidate and add it to the particles Collection
      reco::PFCandidate FakeNeutralHadron(0, momentum,  reco::PFCandidate::h0);
      pOutputCandidateCollection->push_back(FakeNeutralHadron);
      
    }
  }
  
  iEvent.put(pOutputCandidateCollection);
}

DEFINE_FWK_MODULE (FSPFProducer);
