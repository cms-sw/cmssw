#include "DQMOffline/PFTau/interface/METBenchmark.h"

#include "DataFormats/METReco/interface/MET.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



METBenchmark::~METBenchmark() {}


void METBenchmark::setup() {

  PhaseSpace ptPS(100,0,200);
  PhaseSpace phiPS(50, -3.1416, 3.1416);
  PhaseSpace sumEtPS(100, 0, 3000);

  switch(mode_) {
  case DQMOFFLINE:
    ptPS = PhaseSpace(100, 0, 3000);
    break;
  default:
    break;
  }

  pt_ = book1D("pt_", "pt_;p_{T} (GeV)", ptPS.n, ptPS.m, ptPS.M);

  // might want to increase the number of bins, to match the size of the ECAL crystals
  phi_ = book1D("phi_", "phi_;#phi", phiPS.n, phiPS.m, phiPS.M);
  sumEt_ = book1D("sumEt_", "sumEt_;#sumE_{T}", sumEtPS.n, sumEtPS.m, sumEtPS.M);
}


void METBenchmark::fillOne(const reco::MET& cand) {

  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  pt_->Fill( cand.pt() );
  phi_->Fill( cand.phi() );
  sumEt_->Fill( cand.sumEt() );
}
