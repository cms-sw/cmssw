#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



CandidateBenchmark::~CandidateBenchmark() {}


void CandidateBenchmark::setup() {

  cout<<"mode "<<mode_<<endl;

  PhaseSpace ptPS(100,0,100);
  PhaseSpace phiPS(360, -3.1416, 3.1416);
  PhaseSpace etaPS(100, -5,5);
  switch(mode_) {
  case DQMOFFLINE:
  default:
    ptPS = PhaseSpace(50, 0, 100);
    phiPS.n = 50;
    etaPS.n = 20;
    break;
  }

  pt_ = book1D("pt_", "pt_;p_{T} (GeV)", ptPS.n, ptPS.m, ptPS.M);

  eta_ = book1D("eta_", "eta_;#eta", etaPS.n, etaPS.m, etaPS.M);

  // might want to increase the number of bins, to match the size of the ECAL crystals
  phi_ = book1D("phi_", "phi_;#phi", phiPS.n, phiPS.m, phiPS.M);

  charge_ = book1D("charge_", "charge_;charge", 3,-1.5,1.5);
}


void CandidateBenchmark::fillOne(const reco::Candidate& cand) {

  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  pt_->Fill( cand.pt() );
  eta_->Fill( cand.eta() );
  phi_->Fill( cand.phi() );
  charge_->Fill( cand.charge() );
}
