#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



MatchCandidateBenchmark::~MatchCandidateBenchmark() {}


void MatchCandidateBenchmark::setup() {

  PhaseSpace ptPS;
  PhaseSpace dptOvptPS;
  PhaseSpace dptPS;
  PhaseSpace detaPS;
  PhaseSpace dphiPS;
  switch(mode_) {
  case VALIDATION:
    ptPS = PhaseSpace(100,0,1000);
    dptOvptPS = PhaseSpace( 200, -1, 1);
    dphiPS = PhaseSpace( 200, -1, 1);
    detaPS = PhaseSpace( 200, -1, 1);
    dptPS = PhaseSpace( 100, -100, 100);
    break;
  case DQMOFFLINE:
  default:
    ptPS = PhaseSpace(50,0,100);
    dptOvptPS = PhaseSpace( 50, -1, 1);
    dphiPS = PhaseSpace( 50, -1, 1);
    detaPS = PhaseSpace( 50, -1, 1);
    dptPS = PhaseSpace( 50, -50, 50);
    break;
  }

  float ptBins[11] = {0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 1000};

  delta_et_Over_et_VS_et_ = book2D("delta_et_Over_et_VS_et_", 
				   ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				   10, ptBins, 
				   dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );
  

  delta_et_VS_et_ = book2D("delta_et_VS_et_", 
			   ";E_{T, true} (GeV);#DeltaE_{T}",
			   10, ptBins,
			   dptPS.n, dptPS.m, dptPS.M );
  
  delta_eta_VS_et_ = book2D("delta_eta_VS_et_", 
			    ";#E_{T, true} (GeV);#Delta#eta",
			    10, ptBins,
			    detaPS.n, detaPS.m, detaPS.M );
  
  delta_phi_VS_et_ = book2D("delta_phi_VS_et_", 
			    ";E_{T, true} (GeV);#Delta#phi",
			    10, ptBins,
			    dphiPS.n, dphiPS.m, dphiPS.M );
  
  
}



void MatchCandidateBenchmark::fillOne(const reco::Candidate& cand,
				      const reco::Candidate& matchedCand) {
  
  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;
  
  delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
  delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
  delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
  delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );

  
}
