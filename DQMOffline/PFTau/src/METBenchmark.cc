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

  //std::cout << "FL: METBenchmark.cc: start setup()" << std::endl;

  PhaseSpace ptPS(100,0,200);
  PhaseSpace pt2PS(100,0,7000);
  PhaseSpace pxPS(100,-100.,100);
  PhaseSpace phiPS(50, -3.1416, 3.1416);
  PhaseSpace sumEtPS(100, 0, 3000);
  PhaseSpace sumEt2PS(100, 0, 7000);
  PhaseSpace sumEt3PS(50, 0, 200);
  PhaseSpace etOverSumEtPS(100, 0.0, 1.0);

  switch(mode_) {
  case DQMOFFLINE:
    ptPS = PhaseSpace(200, 0, 200);
    pxPS = PhaseSpace(200,-100.,100);
    sumEtPS = PhaseSpace(200, 0, 200);
    break;
  default:
    break;
  }

  pt_ = book1D("pt_", "pt_;p_{T} [GeV]", ptPS.n, ptPS.m, ptPS.M);
  pt2_ = book1D("pt2_", "pt2_;p_{T} [GeV]", pt2PS.n, pt2PS.m, pt2PS.M);
  px_ = book1D("px_", "px_;p_{X} [GeV]", pxPS.n, pxPS.m, pxPS.M);
  py_ = book1D("py_", "py_;p_{Y} [GeV]", pxPS.n, pxPS.m, pxPS.M);

  // might want to increase the number of bins, to match the size of the ECAL crystals
  phi_ = book1D("phi_", "phi_;#phi", phiPS.n, phiPS.m, phiPS.M);
  sumEt_ = book1D("sumEt_", "sumEt_;#SigmaE_{T} [GeV]", sumEtPS.n, sumEtPS.m, sumEtPS.M);
  sumEt2_ = book1D("sumEt2_", "sumEt2_;#SigmaE_{T} [GeV]", sumEt2PS.n, sumEt2PS.m, sumEt2PS.M);
  etOverSumEt_ = book1D("etOverSumEt_", "etOverSumEt_;p_{T}/#SigmaE_{T}", etOverSumEtPS.n, etOverSumEtPS.m, etOverSumEtPS.M);

  mex_VS_sumEt_= book2D("mex_VS_sumEt_", 
			";#SigmaE_{T} [GeV];p_{X} [GeV]",
			sumEt3PS.n, sumEt3PS.m, sumEt3PS.M,
			pxPS.n, pxPS.m, pxPS.M );
}


void METBenchmark::fillOne(const reco::MET& cand) {

  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  pt_->Fill( cand.pt() );
  pt2_->Fill( cand.pt() );
  px_->Fill( cand.px() );
  py_->Fill( cand.py() );
  phi_->Fill( cand.phi() );
  sumEt_->Fill( cand.sumEt() );
  sumEt2_->Fill( cand.sumEt() );
  if (cand.sumEt()>3.0) etOverSumEt_->Fill( cand.pt()/cand.sumEt() );
  mex_VS_sumEt_->Fill( cand.sumEt(), cand.px() );
  mex_VS_sumEt_->Fill( cand.sumEt(), cand.py() );
}
