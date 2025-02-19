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
  
  par1       = iConfig.getParameter<double>("par1");
  par2       = iConfig.getParameter<double>("par2");
  barrel_th  = iConfig.getParameter<double>("barrel_th");
  endcap_th  = iConfig.getParameter<double>("endcap_th");
  middle_th  = iConfig.getParameter<double>("middle_th");
  // register products
  produces<reco::PFCandidateCollection>();
  
}

FSPFProducer::~FSPFProducer() {}

void 
FSPFProducer::beginJob() {}

void 
FSPFProducer::beginRun(edm::Run & iRun,
		       const edm::EventSetup & iSetup) {}

void 
FSPFProducer::produce(Event& iEvent,
		      const EventSetup& iSetup) {
  
  Handle < reco::PFCandidateCollection > pfCandidates;
  iEvent.getByLabel (labelPFCandidateCollection_, pfCandidates);
  
  auto_ptr< reco::PFCandidateCollection >  pOutputCandidateCollection(new PFCandidateCollection);   
  
  LogDebug("FSPFProducer")<<"START event: "
			  <<iEvent.id().event()
			  <<" in run "<<iEvent.id().run()<<endl;   
  
  double theNeutralFraction, px, py, pz, en;
  reco::PFCandidateCollection::const_iterator  itCand =  pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator  itCandEnd = pfCandidates->end();
  for( ; itCand != itCandEnd; itCand++) {
    pOutputCandidateCollection->push_back(*itCand);
    if(itCand->particleId() == reco::PFCandidate::h){
      theNeutralFraction = par1 - par2*itCand->energy();
      if(theNeutralFraction > 0.){
	px = theNeutralFraction*itCand->px();
	py = theNeutralFraction*itCand->py();
	pz = theNeutralFraction*itCand->pz();
	en = sqrt(px*px + py*py + pz*pz); 
	if (en > energy_threshold(itCand->eta())) {
	  // create a PFCandidate and add it to the particles Collection
	  math::XYZTLorentzVector momentum(px,py,pz,en);
	  reco::PFCandidate FakeNeutralHadron(0, momentum,  reco::PFCandidate::h0);
	  pOutputCandidateCollection->push_back(FakeNeutralHadron);
	}
      }
    }
  }
  iEvent.put(pOutputCandidateCollection);
}

double FSPFProducer::energy_threshold(double eta) {
  if (eta<0) eta = -eta;
  if      (eta < 1.6) return barrel_th;
  else if (eta < 1.8) return middle_th;
  else                return endcap_th;
}



DEFINE_FWK_MODULE (FSPFProducer);
