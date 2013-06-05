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
  
  EM_HF_Fraction = iConfig.getParameter< std::vector <double> >("EM_HF_Fraction");
  HF_Ratio   = iConfig.getParameter<double>("HF_Ratio");
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
FSPFProducer::produce(Event& iEvent,
		      const EventSetup& iSetup) {
  
  Handle < reco::PFCandidateCollection > pfCandidates;
  iEvent.getByLabel (labelPFCandidateCollection_, pfCandidates);
  
  auto_ptr< reco::PFCandidateCollection >  pOutputCandidateCollection(new PFCandidateCollection);   
  
  LogDebug("FSPFProducer")<<"START event: "
			  <<iEvent.id().event()
			  <<" in run "<<iEvent.id().run()<<endl;   
  
  double theNeutralFraction, px, py, pz, en;
  float hadron_HF = 0; 
  float em_HF = 0;
  int vEta;
  reco::PFCandidateCollection::const_iterator  itCand =  pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator  itCandEnd = pfCandidates->end();
  for( ; itCand != itCandEnd; itCand++) {
    //pOutputCandidateCollection->push_back(*itCand);
    px = 0.; py = 0.; pz = 0.; en = 0.;theNeutralFraction = 0.;
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
    
    px = 0.; py = 0.; pz = 0.; en = 0.; vEta = -2;
 
    if(itCand->particleId() == reco::PFCandidate::egamma_HF){
      em_HF++;
      
      if(fabs(itCand->eta())< 4.) vEta = 0;
      else if(fabs(itCand->eta())< 5.) vEta = 1;

      px = EM_HF_Fraction[vEta]*itCand->px();
      py = EM_HF_Fraction[vEta]*itCand->py();
      pz = EM_HF_Fraction[vEta]*itCand->pz();
      en = sqrt(px*px + py*py + pz*pz); 
      // create a PFCandidate and add it to the particles Collection
      math::XYZTLorentzVector momentum(px,py,pz,en);
      reco::PFCandidate EMHF(itCand->charge(), momentum,  reco::PFCandidate::egamma_HF);
      if(en>0.) pOutputCandidateCollection->push_back(EMHF);
    }
    
    else if(itCand->particleId() == reco::PFCandidate::h_HF){
      hadron_HF++;
      if(em_HF < (hadron_HF*HF_Ratio)) pOutputCandidateCollection->push_back(*itCand);
    }
    else  pOutputCandidateCollection->push_back(*itCand);
    
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
