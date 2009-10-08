#include "RecoParticleFlow/Benchmark/interface/CandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;


CandidateBenchmark::CandidateBenchmark() {}

CandidateBenchmark::~CandidateBenchmark() {}


void CandidateBenchmark::setup() {

  pt_ = book1D("pt_", "pt_;p_{T} (GeV)", 100, 0, 100);

  eta_ = book1D("eta_", "eta_", 100,-5,5);

  // might want to increase the number of bins, to match the size of the ECAL crystals
  phi_ = book1D("phi_", "phi_", 100,-3.2,3.2);

  charge_ = book1D("charge_", "charge_", 3,-1.5,1.5);
}


void CandidateBenchmark::fill(const Collection& candCollection ) {
  
  for (unsigned int i = 0; i < candCollection.size(); i++) {
    const reco::Candidate& cand = candCollection[i];
    fill(cand);
  }
}


void CandidateBenchmark::fill(const reco::Candidate& cand) {

  pt_->Fill( cand.pt() );
  eta_->Fill( cand.eta() );
  phi_->Fill( cand.phi() );
  charge_->Fill( cand.charge() );
}
