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
  
  pt_gen_   = nullptr;
  eta_gen_  = nullptr;
  phi_gen_  = nullptr;

  pt_ref_   = nullptr;
  eta_ref_  = nullptr;
  phi_ref_  = nullptr;

  deltaR_   = nullptr;

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

  dRMax_                  = parameterSet.getParameter<double>( "deltaRMax" );
  matchCharge_            = parameterSet.getParameter<bool>( "matchCharge" );
  mode_                   = (Benchmark::Mode) parameterSet.getParameter<int>( "mode" );
  createReferenceHistos_  = parameterSet.getParameter<bool>( "CreateReferenceHistos" );
  createEfficiencyHistos_ = parameterSet.getParameter<bool>( "CreateEfficiencyHistos" );
  
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
void PFCandidateMonitor::setup(DQMStore::IBooker& b, const edm::ParameterSet & parameterSet) {
  candBench_.setup(b, parameterSet);
  matchCandBench_.setup(b, parameterSet);

  if (createReferenceHistos_ && !histogramBooked_) {
    edm::ParameterSet ptPS  = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    edm::ParameterSet etaPS = parameterSet.getParameter<edm::ParameterSet>("EtaHistoParameter");
    edm::ParameterSet phiPS = parameterSet.getParameter<edm::ParameterSet>("PhiHistoParameter");

    edm::ParameterSet dR = parameterSet.getParameter<edm::ParameterSet>("DeltaRHistoParameter");

    if (ptPS.getParameter<bool>("switchOn")) {
      pt_ref_ = book1D(b, "pt_ref_", "p_{T}_ref;p_{T} (GeV)", ptPS.getParameter<int32_t>("nBin"), 
		       ptPS.getParameter<double>("xMin"),
		       ptPS.getParameter<double>("xMax"));
      if (createEfficiencyHistos_) {
	pt_gen_ = book1D(b, "pt_gen_", "p_{T}_gen;p_{T} (GeV)", ptPS.getParameter<int32_t>("nBin"), 
			 ptPS.getParameter<double>("xMin"),
			 ptPS.getParameter<double>("xMax") ) ;
      }
    } 
    
    if (etaPS.getParameter<bool>("switchOn")) {
      eta_ref_ = book1D(b, "eta_ref_", "#eta_ref;#eta", etaPS.getParameter<int32_t>("nBin"), 
			etaPS.getParameter<double>("xMin"),
			etaPS.getParameter<double>("xMax"));
      if (createEfficiencyHistos_) {
	eta_gen_ = book1D(b, "eta_gen_", "#eta_gen;#eta", etaPS.getParameter<int32_t>("nBin"), 
			  etaPS.getParameter<double>("xMin"),
			  etaPS.getParameter<double>("xMax") ) ;
      }
    }

    if (phiPS.getParameter<bool>("switchOn")) {
      phi_ref_ = book1D(b, "phi_ref_", "#phi_ref;#phi", phiPS.getParameter<int32_t>("nBin"), 
			phiPS.getParameter<double>("xMin"),
			phiPS.getParameter<double>("xMax"));
      if (createEfficiencyHistos_) {
	phi_gen_ = book1D(b, "phi_gen_", "#phi_gen;#phi", phiPS.getParameter<int32_t>("nBin"), 
			  phiPS.getParameter<double>("xMin"),
			  phiPS.getParameter<double>("xMax") ) ;
      }
    }

    if ( createEfficiencyHistos_ && dR.getParameter<bool>("switchOn") ) { 
      deltaR_ = book1D(b, "deltaR_", "#DeltaR;#DeltaR",
		       dR.getParameter<int32_t>("nBin"), 
		       dR.getParameter<double>("xMin"),
		       dR.getParameter<double>("xMax"));
    }

    histogramBooked_ = true;   
  }
}


//
// -- Create histograms using local parameters
//
void PFCandidateMonitor::setup(DQMStore::IBooker& b) {
  candBench_.setup(b);
  matchCandBench_.setup(b);

  if (createReferenceHistos_ && !histogramBooked_) {
    PhaseSpace ptPS(100,0,100);
    PhaseSpace phiPS(360, -3.1416, 3.1416);
    PhaseSpace etaPS(100, -5,5);
    
    pt_ref_ = book1D(b, "pt_ref_", "p_{T}_ref;p_{T} (GeV)", ptPS.n, ptPS.m, ptPS.M);
    if (createEfficiencyHistos_) {
      pt_gen_ = book1D(b, "pt_gen_", "p_{T}_gen;p_{T} (GeV)", ptPS.n, ptPS.m, ptPS.M);
    }

    eta_ref_ = book1D(b, "eta_ref_", "#eta_ref;#eta", etaPS.n, etaPS.m, etaPS.M);
    if (createEfficiencyHistos_) {
      eta_gen_ = book1D(b, "eta_gen_", "#eta_gen;#eta", etaPS.n, etaPS.m, etaPS.M);
    }

    phi_ref_ = book1D(b, "phi_ref_", "#phi_ref;#phi", phiPS.n, phiPS.m, phiPS.M);
    if (createEfficiencyHistos_) {
      phi_gen_ = book1D(b, "phi_gen_", "#phi_gen;#phi", phiPS.n, phiPS.m, phiPS.M);
    }

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
// -- fill histograms for a single collection
//
void PFCandidateMonitor::fillOne(const reco::Candidate& cand) {

  if (matching_done_) {
    if (createReferenceHistos_ && histogramBooked_) {
      if (pt_ref_) pt_ref_->Fill( cand.pt() );
      if (eta_ref_) eta_ref_->Fill( cand.eta() );
      if (phi_ref_) phi_ref_->Fill( cand.phi() );
    }
  } else if (createEfficiencyHistos_ && histogramBooked_) {
    if (pt_gen_) pt_gen_->Fill( cand.pt() );
    if (eta_gen_) eta_gen_->Fill( cand.eta() );
    if (phi_gen_) phi_gen_->Fill( cand.phi() );
  }

}
