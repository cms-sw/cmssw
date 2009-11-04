#include "DQMOffline/PFTau/interface/PFCandidateBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



PFCandidateBenchmark::~PFCandidateBenchmark() {}


void PFCandidateBenchmark::setup() {

  PhaseSpace ecalEnergyPS(100,0,100);
  PhaseSpace hcalEnergyPS(100,0,100);
  PhaseSpace mva_e_piPS(100,-1,1);
  switch(mode_) {
  case VALIDATION:
    break;
  case DQMOFFLINE:
  default:
    ecalEnergyPS.n = 50;
    hcalEnergyPS.n = 50;
    mva_e_piPS.n = 50;
    break;
    break;
  }

  particleId_ = book1D("particleId_", "particle ID", 7,1,8);
  ecalEnergy_ = book1D("ecalEnergy_", "ECAL energy, corrected;E_{ECAL} (GeV)",
		       ecalEnergyPS.n, ecalEnergyPS.m, ecalEnergyPS.M);
  hcalEnergy_ = book1D("hcalEnergy_", "HCAL energy, corrected;E_{HCAL} (GeV)",
		       ecalEnergyPS.n, ecalEnergyPS.m, ecalEnergyPS.M);
  mva_e_pi_ = book1D("mva_e_pi_", "e VS #pi MVA output;MVA", 
		     mva_e_piPS.n, mva_e_piPS.m, mva_e_piPS.M);
  elementsInBlocksSize_ = book1D("elementsInBlocksSize_", "number of elements used", 10, 0, 10);
}



void PFCandidateBenchmark::fill( const reco::PFCandidateCollection& pfCands) {

  for(unsigned i=0; i<pfCands.size(); ++i) {
    fillOne(pfCands[i]);
  }
}


void PFCandidateBenchmark::fillOne( const reco::PFCandidate& pfCand ) {

  if( !isInRange(pfCand.pt(), pfCand.eta(), pfCand.phi() ) ) return;

  particleId_->Fill( pfCand.particleId() );
  ecalEnergy_->Fill( pfCand.ecalEnergy() );
  hcalEnergy_->Fill( pfCand.hcalEnergy() );
  mva_e_pi_->Fill( pfCand.mva_e_pi() );
  elementsInBlocksSize_->Fill( pfCand.elementsInBlocks().size() );
}
