#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Matchers.h"

#include "DQMOffline/PFTau/interface/PFCandidateMonitor.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
//
// -- Constructor
//
PFCandidateMonitor::PFCandidateMonitor( float dRMax, bool matchCharge, Benchmark::Mode mode) : 
  Benchmark(mode), 
  candBench_(mode), 
  matchCandBench_(mode), 
  dRMax_(dRMax), 
  matchCharge_(matchCharge) {
  
  setRange( 0.0, 10e10, -10.0, 10.0, -3.14, 3.14);
  
  pt_ref_   = 0;
  eta_ref_  = 0;
  phi_ref_  = 0;
  
  createReferenceHistos_ = false;
  histogramBooked_ = false;
}  
//
// -- Destructor
//
PFCandidateMonitor::~PFCandidateMonitor() {}

//
// -- Set Parameters accessing them from ParameterSet
//
void PFCandidateMonitor::setParameters( const edm::ParameterSet & parameterSet) {

  dRMax_                 = parameterSet.getParameter<double>( "deltaRMax" );
  matchCharge_           = parameterSet.getParameter<bool>( "matchCharge" );
  mode_                  = (Benchmark::Mode) parameterSet.getParameter<int>( "mode" );
  createReferenceHistos_ = parameterSet.getParameter<bool>( "CreateReferenceHistos" );

  
  setRange( parameterSet.getParameter<double>("ptMin"),
	    parameterSet.getParameter<double>("ptMax"),
	    parameterSet.getParameter<double>("etaMin"),
	    parameterSet.getParameter<double>("etaMax"),
	    parameterSet.getParameter<double>("phiMin"),
	    parameterSet.getParameter<double>("phiMax") );

  candBench_.setParameters(mode_);
  matchCandBench_.setParameters(mode_);
}
//
// -- Set Parameters 
//
void PFCandidateMonitor::setParameters(float dRMax, bool matchCharge, Benchmark::Mode mode,
				 float ptmin, float ptmax, float etamin, float etamax, 
				 float phimin, float phimax, bool refHistoFlag) {
  dRMax_                 = dRMax;
  matchCharge_           = matchCharge;
  mode_                  = mode;
  createReferenceHistos_ = refHistoFlag;
  
  setRange( ptmin, ptmax, etamin, etamax, phimin, phimax );

  candBench_.setParameters(mode_);
  matchCandBench_.setParameters(mode_);
}
//
// -- Create histograms accessing parameters from ParameterSet
//
void PFCandidateMonitor::setup(const edm::ParameterSet & parameterSet) {
  candBench_.setup(parameterSet);
  matchCandBench_.setup(parameterSet);

  if (createReferenceHistos_ && !histogramBooked_) {
    edm::ParameterSet ptPS  = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    edm::ParameterSet etaPS = parameterSet.getParameter<edm::ParameterSet>("EtaHistoParameter");
    edm::ParameterSet phiPS = parameterSet.getParameter<edm::ParameterSet>("PhiHistoParameter");
    if (ptPS.getParameter<bool>("switchOn")) {
      pt_ref_ = book1D("pt_ref_", "pt_ref_;p_{T} (GeV)", ptPS.getParameter<int32_t>("nBin"), 
		   ptPS.getParameter<double>("xMin"),
		   ptPS.getParameter<double>("xMax"));
    } 
    
    if (etaPS.getParameter<bool>("switchOn")) {
      eta_ref_ = book1D("eta_ref_", "eta_ref_;#eta_ref_", etaPS.getParameter<int32_t>("nBin"), 
		    etaPS.getParameter<double>("xMin"),
		    etaPS.getParameter<double>("xMax"));
    }
    if (phiPS.getParameter<bool>("switchOn")) {
      phi_ref_ = book1D("phi_ref_", "phi_ref_;#phref_i", phiPS.getParameter<int32_t>("nBin"), 
		    phiPS.getParameter<double>("xMin"),
		    phiPS.getParameter<double>("xMax"));
    }
    histogramBooked_ = true;   
  }
}
//
// -- Create histograms using local parameters
//
void PFCandidateMonitor::setup() {
  candBench_.setup();
  matchCandBench_.setup();

  if (createReferenceHistos_ && !histogramBooked_) {
    PhaseSpace ptPS(100,0,100);
    PhaseSpace phiPS(360, -3.1416, 3.1416);
    PhaseSpace etaPS(100, -5,5);
    
    pt_ref_ = book1D("pt_ref_", "pt_ref_;p_{T} (GeV)", ptPS.n, ptPS.m, ptPS.M);
    
    eta_ref_ = book1D("eta_ref_", "eta_ref_;#eta", etaPS.n, etaPS.m, etaPS.M);
    
    phi_ref_ = book1D("phi_ref_", "phi_ref_;#phi", phiPS.n, phiPS.m, phiPS.M);

    histogramBooked_ = true;
  }
}
//
// -- Set directory to book histograms using ROOT
//
void PFCandidateMonitor::setDirectory(TDirectory* dir) {
  Benchmark::setDirectory(dir);

  candBench_.setDirectory(dir);
  matchCandBench_.setDirectory(dir);
}
//
// -- fill histograms for a given Jet pair
//
void PFCandidateMonitor::fillOne(const reco::Candidate& cand) {

  if (createReferenceHistos_ && histogramBooked_) {
 
    if (pt_ref_) pt_ref_->Fill(cand.pt());
    if (eta_ref_) eta_ref_->Fill(cand.eta() );
    if (phi_ref_) phi_ref_->Fill(cand.phi() );
  }
}

