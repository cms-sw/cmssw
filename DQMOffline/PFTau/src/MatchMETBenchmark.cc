#include "DQMOffline/PFTau/interface/MatchMETBenchmark.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

using namespace std;

MatchMETBenchmark::~MatchMETBenchmark() {}

void MatchMETBenchmark::setup() {

  //std::cout << "FL: MatchMETBenchmark.cc: start setup()" << std::endl;
  PhaseSpace ptPS;
  PhaseSpace dptOvptPS;
  PhaseSpace dptPS;
  PhaseSpace dphiPS;
  PhaseSpace setPS;
  PhaseSpace dsetPS;
  PhaseSpace setOvsetPS;

  switch(mode_) {
  case VALIDATION:
    ptPS = PhaseSpace(100,0,1000);
    dptOvptPS = PhaseSpace( 200, -1, 1);
    dphiPS = PhaseSpace( 100, -3.2, 3.2);
    dptPS = PhaseSpace( 200, -500, 500);
    setPS = PhaseSpace( 300, 0.0, 3000);
    dsetPS = PhaseSpace( 200, 0.-1000, 1000);
    setOvsetPS = PhaseSpace( 500,0., 2.);
    break;
  case DQMOFFLINE:
  default:
    ptPS = PhaseSpace(50,0,200);
    dptOvptPS = PhaseSpace( 50, -1, 1);
    dphiPS = PhaseSpace( 50, -3.2, 3.2);
    dptPS = PhaseSpace( 50, -500, 500);
    setPS = PhaseSpace( 50, 0.0, 3000);
    dsetPS = PhaseSpace( 50, -1000.0, 1000);
    setOvsetPS = PhaseSpace( 100,0., 2.);
    break;
  }

  // variable bins to be done here, as they will save a lot of memory. 

  //float ptBins[11] = {0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 1000};

  delta_et_Over_et_VS_et_ = book2D("delta_et_Over_et_VS_et_", 
				   ";ME_{T, true} (GeV);#DeltaME_{T}/ME_{T}",
				   ptPS.n, ptPS.m, ptPS.M,
				   dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );

  delta_et_VS_et_ = book2D("delta_et_VS_et_", 
			   ";ME_{T, true} (GeV);#DeltaME_{T}",
			   ptPS.n, ptPS.m, ptPS.M,
			   dptPS.n, dptPS.m, dptPS.M );

  delta_phi_VS_et_ = book2D("delta_phi_VS_et_", 
			    ";ME_{T, true} (GeV);#Delta#phi",
			    ptPS.n, ptPS.m, ptPS.M,
			    dphiPS.n, dphiPS.m, dphiPS.M );

  delta_ex_ = book1D("delta_ex_", 
			   "#DeltaME_{X}",
			   dptPS.n, dptPS.m, dptPS.M );

  RecEt_VS_TrueEt_ = book2D("RecEt_VS_TrueEt_", 
			   ";ME_{T, true} (GeV);ME_{T}",
			   ptPS.n, ptPS.m, ptPS.M,
			   ptPS.n, ptPS.m, ptPS.M );

  delta_set_VS_set_ = book2D("delta_set_VS_set_", 
			   ";SE_{T, true} (GeV);#DeltaSE_{T}",
			   setPS.n, setPS.m, setPS.M,
			   dsetPS.n, dsetPS.m, dsetPS.M );

  delta_set_Over_set_VS_set_ = book2D("delta_set_Over_set_VS_set_", 
			   ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
			   setPS.n, setPS.m, setPS.M,
			   dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );

  delta_ex_VS_set_ = book2D("delta_ex_VS_set_", 
			   ";SE_{T, true} (GeV);#DeltaE_{X}",
			   setPS.n, setPS.m, setPS.M,
			   ptPS.n, -ptPS.M, ptPS.M );

  RecSet_Over_TrueSet_VS_TrueSet_ = book2D("RecSet_Over_TrueSet_VS_TrueSet_", 
			   ";SE_{T, true} (GeV);SE_{T}/SE_{T}",
			   setPS.n, setPS.m, setPS.M,
			   setOvsetPS.n, setOvsetPS.m, setOvsetPS.M );
}

void MatchMETBenchmark::fillOne(const reco::MET& cand,
				      const reco::MET& matchedCand) {
  
  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;
  
  if ( matchedCand.pt()>0.001 ) delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
  else edm::LogWarning("MatchMETBenchmark") << " matchedCand.pt()<0.001";
  delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
  delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
  delta_ex_->Fill(cand.px()-matchedCand.px());
  delta_ex_->Fill(cand.py()-matchedCand.py());
  RecEt_VS_TrueEt_->Fill(matchedCand.pt(),cand.pt());
  delta_set_VS_set_->Fill(matchedCand.sumEt(),cand.sumEt()-matchedCand.sumEt());
  if ( matchedCand.sumEt()>0.001 ) delta_set_Over_set_VS_set_->Fill(matchedCand.sumEt(),(cand.sumEt()-matchedCand.sumEt())/matchedCand.sumEt());
  else edm::LogWarning("MatchMETBenchmark") << " matchedCand.sumEt()<0.001";
  delta_ex_VS_set_->Fill(matchedCand.sumEt(),cand.px()-matchedCand.px());
  delta_ex_VS_set_->Fill(matchedCand.sumEt(),cand.py()-matchedCand.py());
  if ( matchedCand.sumEt()>0.001 ) RecSet_Over_TrueSet_VS_TrueSet_->Fill(matchedCand.sumEt(),cand.sumEt()/matchedCand.sumEt());

}
