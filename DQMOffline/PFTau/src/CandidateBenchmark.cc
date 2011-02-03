#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;


CandidateBenchmark::CandidateBenchmark(Mode mode) : Benchmark(mode) {
  pt_     = 0; 
  eta_    = 0; 
  phi_    = 0; 
  charge_ = 0;
  pdgId_  = 0;

  histogramBooked_ = false;
 
}

CandidateBenchmark::~CandidateBenchmark() {}


void CandidateBenchmark::setup() {

  if (!histogramBooked_) {
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
    
    phi_ = book1D("phi_", "phi_;#phi", phiPS.n, phiPS.m, phiPS.M);

    charge_ = book1D("charge_", "charge_;charge", 3,-1.5,1.5);

    histogramBooked_ = true;
  }
}

void CandidateBenchmark::setup(const edm::ParameterSet& parameterSet) {
  if (!histogramBooked_)  {
    edm::ParameterSet ptPS  = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    edm::ParameterSet etaPS = parameterSet.getParameter<edm::ParameterSet>("EtaHistoParameter");
    edm::ParameterSet phiPS = parameterSet.getParameter<edm::ParameterSet>("PhiHistoParameter");
    edm::ParameterSet chPS = parameterSet.getParameter<edm::ParameterSet>("ChargeHistoParameter");
    
    if (ptPS.getParameter<bool>("switchOn")) {
      pt_ = book1D("pt_", "pt_;p_{T} (GeV)", ptPS.getParameter<int32_t>("nBin"), 
		   ptPS.getParameter<double>("xMin"),
		   ptPS.getParameter<double>("xMax"));
    } 
    
    if (etaPS.getParameter<bool>("switchOn")) {
      eta_ = book1D("eta_", "eta_;#eta", etaPS.getParameter<int32_t>("nBin"), 
		    etaPS.getParameter<double>("xMin"),
		    etaPS.getParameter<double>("xMax"));
    }
    if (phiPS.getParameter<bool>("switchOn")) {
      phi_ = book1D("phi_", "phi_;#phi", phiPS.getParameter<int32_t>("nBin"), 
		    phiPS.getParameter<double>("xMin"),
		    phiPS.getParameter<double>("xMax"));
    }
    if (chPS.getParameter<bool>("switchOn")) {
      charge_ = book1D("charge_","charge_;charge",chPS.getParameter<int32_t>("nBin"),
		       chPS.getParameter<double>("xMin"),
		       chPS.getParameter<double>("xMax"));
    }   
    histogramBooked_ = true;
  }
}

void CandidateBenchmark::fillOne(const reco::Candidate& cand) {

  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  if (histogramBooked_) {
    if (pt_) pt_->Fill( cand.pt() );
    if (eta_) eta_->Fill( cand.eta() );
    if (phi_) phi_->Fill( cand.phi() );
    if (charge_) charge_->Fill( cand.charge() );
  }
}
