#include "DQM/PhysicsHWW/interface/PFElectronMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;
typedef math::XYZPoint Point;
typedef edm::ValueMap<reco::PFCandidatePtr> PFCandMap;

using namespace reco;
using namespace edm;
using namespace std;

PFElectronMaker::PFElectronMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  PFElectrons_ = iCollector.consumes<edm::ValueMap<reco::PFCandidatePtr> >(iConfig.getParameter<edm::InputTag>("pfElectronsTag"));

}

void PFElectronMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  hww.Load_pfels_p4();

  bool validToken;

  Handle<PFCandMap > pfCandidatesHandle;
  validToken = iEvent.getByToken( PFElectrons_, pfCandidatesHandle );
  if(!validToken) return;
  const ValueMap<reco::PFCandidatePtr> *pfCandidates  = pfCandidatesHandle.product();
  
  PFCandMap::const_iterator pf_pit = pfCandidates->begin();
  unsigned int nC = pf_pit.size();
  for( unsigned int iC = 0; iC < nC; ++iC ) {

    const PFCandidatePtr& pf_it = pf_pit[iC];
    if ( pf_it.isNull() ) continue;
    int pfflags = 0;

    for( unsigned int i = 0; i < 17; i++ ) {
      if(pf_it->flag((PFCandidate::Flags)i)) pfflags |= (1<<i);
    }
  
    hww.pfels_p4()               .push_back(LorentzVector(pf_it->px(), pf_it->py(), pf_it->pz(), pf_it->p()) );
    
  } 
}
