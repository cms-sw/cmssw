#include "RecoParticleFlow/Benchmark/interface/PFCandidateBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;


PFCandidateBenchmark::PFCandidateBenchmark() {}

PFCandidateBenchmark::~PFCandidateBenchmark() {}


void PFCandidateBenchmark::setup() {

  CandidateBenchmark::setup();
  particleId_ = book1D("particleId_", "particle ID", 15,0,15);
  ecalEnergy_ = book1D("ecalEnergy_", "ECAL energy, corrected;E_{ECAL} (GeV)", 100, 0, 100);
  hcalEnergy_ = book1D("hcalEnergy_", "HCAL energy, corrected;E_{HCAL} (GeV)",100, 0, 100);
  mva_e_pi_ = book1D("mva_e_pi_", "e VS #pi MVA output;MVA", 500, -1, 1);
  elementsInBlocksSize_ = book1D("elementsInBlocksSize_", "number of elements used", 10, 0, 10);
}


void PFCandidateBenchmark::fill(const Collection& pfCandCollection ) {
  
  for (unsigned int i = 0; i < pfCandCollection.size(); i++) {
    
    const reco::PFCandidate& pfc = pfCandCollection[i];
    fill( pfc );
  }
}


void PFCandidateBenchmark::fill( const reco::PFCandidate& pfCand ) {

  CandidateBenchmark::fill( pfCand );

  // specific histograms:

  particleId_->Fill( pfCand.particleId() );
  ecalEnergy_->Fill( pfCand.ecalEnergy() );
  hcalEnergy_->Fill( pfCand.hcalEnergy() );
  mva_e_pi_->Fill( pfCand.mva_e_pi() );
  elementsInBlocksSize_->Fill( pfCand.elementsInBlocks().size() );
}
