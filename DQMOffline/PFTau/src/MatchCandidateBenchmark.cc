#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;

MatchCandidateBenchmark::MatchCandidateBenchmark(Mode mode)  : Benchmark(mode) {
  delta_et_Over_et_VS_et_ = 0; 
  delta_et_VS_et_         = 0; 
  delta_eta_VS_et_        = 0; 
  delta_phi_VS_et_        = 0;

  histogramBooked_ = false;
}

MatchCandidateBenchmark::~MatchCandidateBenchmark() {}


void MatchCandidateBenchmark::setup() {
  if (!histogramBooked_) {
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
    histogramBooked_ = true;
  } 
}
void MatchCandidateBenchmark::setup(const edm::ParameterSet& parameterSet) {

  if (!histogramBooked_) {
    
    edm::ParameterSet dptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtHistoParameter");
    edm::ParameterSet dptOvptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtOvPtHistoParameter");
    edm::ParameterSet detaPS = parameterSet.getParameter<edm::ParameterSet>("DeltaEtaHistoParameter");
    edm::ParameterSet dphiPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPhiHistoParameter");
    
    std::vector<double> ptBinsPS = parameterSet.getParameter< std::vector<double> >( "VariablePtBins" );
    float* ptBins = new float[ptBinsPS.size()];
    for (size_t i = 0; i < ptBinsPS.size(); i++) {
      ptBins[i] = ptBinsPS[i];
    }
    
    if (dptOvptPS.getParameter<bool>("switchOn")) {
      delta_et_Over_et_VS_et_ = book2D("delta_et_Over_et_VS_et_", 
				       ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				       ptBinsPS.size()-1, ptBins, 
				       dptOvptPS.getParameter<int32_t>("nBin"), 
				       dptOvptPS.getParameter<double>("xMin"), 
				       dptOvptPS.getParameter<double>("xMax"));
    }
    
    if (dptPS.getParameter<bool>("switchOn")) {
      delta_et_VS_et_ = book2D("delta_et_VS_et_", 
			       ";E_{T, true} (GeV);#DeltaE_{T}",
			       ptBinsPS.size()-1, ptBins,
			       dptPS.getParameter<int32_t>("nBin"), 
			       dptPS.getParameter<double>("xMin"), 
			       dptPS.getParameter<double>("xMax"));
    }
    
    if (detaPS.getParameter<bool>("switchOn")) {
      delta_eta_VS_et_ = book2D("delta_eta_VS_et_", 
				";#E_{T, true} (GeV);#Delta#eta",
				ptBinsPS.size()-1, ptBins,
				detaPS.getParameter<int32_t>("nBin"), 
				detaPS.getParameter<double>("xMin"), 
				detaPS.getParameter<double>("xMax"));
    }
    
    if (dphiPS.getParameter<bool>("switchOn")) {
      delta_phi_VS_et_ = book2D("delta_phi_VS_et_", 
				";E_{T, true} (GeV);#Delta#phi",
				ptBinsPS.size()-1, ptBins,
				dphiPS.getParameter<int32_t>("nBin"), 
				dphiPS.getParameter<double>("xMin"),
				dphiPS.getParameter<double>("xMax"));
    }
    histogramBooked_ = true;
    delete ptBins;
  }
}

void MatchCandidateBenchmark::fillOne(const reco::Candidate& cand,
				      const reco::Candidate& matchedCand) {
  
  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  if (histogramBooked_) {
    if (delta_et_Over_et_VS_et_) delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
    if (delta_et_VS_et_) delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
    if (delta_eta_VS_et_) delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
    if (delta_phi_VS_et_) delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
  }  

}
