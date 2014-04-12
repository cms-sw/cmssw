#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include <TProfile.h>

using namespace std;

MatchCandidateBenchmark::MatchCandidateBenchmark(Mode mode)  : Benchmark(mode) {

  delta_et_Over_et_VS_et_   = 0; 
  delta_et_VS_et_           = 0; 
  delta_eta_VS_et_          = 0; 
  delta_phi_VS_et_          = 0;

  BRdelta_et_Over_et_VS_et_ = 0; 
  ERdelta_et_Over_et_VS_et_ = 0; 
  // pTRes are initialzied in the setup since ptBinsPS.size() is needed

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
    int size = sizeof(ptBins)/sizeof(*ptBins);

    delta_et_Over_et_VS_et_ = book2D("delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				     size, ptBins, 
				     dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );
    
    BRdelta_et_Over_et_VS_et_ = book2D("BRdelta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				     size, ptBins, 
				     dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );
    ERdelta_et_Over_et_VS_et_ = book2D("ERdelta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				     size, ptBins, 
				     dptOvptPS.n, dptOvptPS.m, dptOvptPS.M );
    
    delta_et_VS_et_ = book2D("delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
			     size, ptBins,
			     dptPS.n, dptPS.m, dptPS.M );
    
    delta_eta_VS_et_ = book2D("delta_eta_VS_et_", ";#E_{T, true} (GeV);#Delta#eta",
			      size, ptBins,
			      detaPS.n, detaPS.m, detaPS.M );
    
    delta_phi_VS_et_ = book2D("delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
			      size, ptBins,
			      dphiPS.n, dphiPS.m, dphiPS.M );
    /*
    // TProfile
    profile_delta_et_Over_et_VS_et_ = bookProfile("profile_delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
						  size, ptBins, 
						  dptOvptPS.m, dptOvptPS.M, "" );
        
    profile_delta_et_VS_et_ = bookProfile("profile_delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
					  size, ptBins,
					  dptPS.m, dptPS.M, "" );

    profile_delta_eta_VS_et_ = bookProfile("profile_delta_eta_VS_et_", ";#E_{T, true} (GeV);#Delta#eta",
					   size, ptBins,
					   detaPS.m, detaPS.M, "" );
    
    profile_delta_phi_VS_et_ = bookProfile("profile_delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
					   size, ptBins,
					   dphiPS.m, dphiPS.M, "" );
    // TProfile RMS
    profileRMS_delta_et_Over_et_VS_et_ = bookProfile("profileRMS_delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
						  size, ptBins, 
						  dptOvptPS.m, dptOvptPS.M, "s" );
        
    profileRMS_delta_et_VS_et_ = bookProfile("profileRMS_delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
					  size, ptBins,
					  dptPS.m, dptPS.M, "s" );

    profileRMS_delta_eta_VS_et_ = bookProfile("profileRMS_delta_eta_VS_et_", ";#E_{T, true} (GeV);#Delta#eta",
					   size, ptBins,
					   detaPS.m, detaPS.M, "s" );
    
    profileRMS_delta_phi_VS_et_ = bookProfile("profileRMS_delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
					   size, ptBins,
					   dphiPS.m, dphiPS.M, "s" );
    */
    pTRes_.resize(size); BRpTRes_.resize(size); ERpTRes_.resize(size);
    for (size_t i = 0; i < pTRes_.size(); i++) {
      pTRes_[i] = 0; BRpTRes_[i] = 0; ERpTRes_[i] = 0; }

    histogramBooked_ = true;
  } 
}
void MatchCandidateBenchmark::setup(const edm::ParameterSet& parameterSet) {

  std::vector<double> ptBinsPS = parameterSet.getParameter< std::vector<double> >( "VariablePtBins" );
  pTRes_.resize(ptBinsPS.size()-1); BRpTRes_.resize(ptBinsPS.size()-1); ERpTRes_.resize(ptBinsPS.size()-1);
  if (pTRes_.size() > 0) 
    for (size_t i = 0; i < pTRes_.size(); i++) {
      pTRes_[i] = 0; BRpTRes_[i] = 0; ERpTRes_[i] = 0; } 
  
  if (!histogramBooked_) { 
    
    edm::ParameterSet ptPS = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    edm::ParameterSet dptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtHistoParameter");
    edm::ParameterSet dptOvptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtOvPtHistoParameter");
    edm::ParameterSet detaPS = parameterSet.getParameter<edm::ParameterSet>("DeltaEtaHistoParameter");
    edm::ParameterSet dphiPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPhiHistoParameter");

    std::vector<float> ptBins; 
    if (ptBinsPS.size() > 1) { 
      ptBins.reserve(ptBinsPS.size());
      for (size_t i = 0; i < ptBinsPS.size(); i++) 
	ptBins.push_back(ptBinsPS[i]); 
    } else { 
      Int_t nFixedBins = ptPS.getParameter<int32_t>("nBin");
      ptBins.reserve(nFixedBins+1);
      for (Int_t i = 0; i <= nFixedBins; i++) 
	ptBins.push_back(ptPS.getParameter<double>("xMin") + i*((ptPS.getParameter<double>("xMax") - ptPS.getParameter<double>("xMin")) / nFixedBins)) ; 
      ptBinsPS.resize(nFixedBins);
    }
 
   if (dptOvptPS.getParameter<bool>("switchOn")) {
      delta_et_Over_et_VS_et_ = book2D("delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
				       ptBinsPS.size()-1, &(ptBins.front()), 
				       dptOvptPS.getParameter<int32_t>("nBin"), 
				       dptOvptPS.getParameter<double>("xMin"), 
				       dptOvptPS.getParameter<double>("xMax"));
   }
   if (dptOvptPS.getParameter<bool>("slicingOn")) {
     for (size_t i = 0; i < pTRes_.size(); i++) {
       pTRes_[i] = book1D( TString::Format("Pt%d_%d", (int)ptBins[i], (int)ptBins[i+1]), ";#Deltap_{T}/p_{T};Entries",
			   dptOvptPS.getParameter<int32_t>("nBin"), 
			   dptOvptPS.getParameter<double>("xMin"), 
			   dptOvptPS.getParameter<double>("xMax")); 
       BRpTRes_[i] = book1D( TString::Format("BRPt%d_%d", (int)ptBins[i], (int)ptBins[i+1]), ";#Deltap_{T}/p_{T};Entries",
			     dptOvptPS.getParameter<int32_t>("nBin"), 
			     dptOvptPS.getParameter<double>("xMin"), 
			     dptOvptPS.getParameter<double>("xMax")); 
       ERpTRes_[i] = book1D( TString::Format("ERPt%d_%d", (int)ptBins[i], (int)ptBins[i+1]), ";#Deltap_{T}/p_{T};Entries",
			     dptOvptPS.getParameter<int32_t>("nBin"), 
			     dptOvptPS.getParameter<double>("xMin"), 
			     dptOvptPS.getParameter<double>("xMax")); 
     }
   }
   if (dptOvptPS.getParameter<bool>("BROn")) {
     BRdelta_et_Over_et_VS_et_ = book2D("BRdelta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
					ptBinsPS.size()-1, &(ptBins.front()), 
					dptOvptPS.getParameter<int32_t>("nBin"), 
					dptOvptPS.getParameter<double>("xMin"), 
					dptOvptPS.getParameter<double>("xMax"));
   }
    if (dptOvptPS.getParameter<bool>("EROn")) {
      ERdelta_et_Over_et_VS_et_ = book2D("ERdelta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
					 ptBinsPS.size()-1, &(ptBins.front()), 
					 dptOvptPS.getParameter<int32_t>("nBin"), 
					 dptOvptPS.getParameter<double>("xMin"), 
					 dptOvptPS.getParameter<double>("xMax"));
    }
    
    if (dptPS.getParameter<bool>("switchOn")) {
      delta_et_VS_et_ = book2D("delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
			       ptBinsPS.size()-1, &(ptBins.front()),
			       dptPS.getParameter<int32_t>("nBin"), 
			       dptPS.getParameter<double>("xMin"), 
			       dptPS.getParameter<double>("xMax"));
    }
    
    if (detaPS.getParameter<bool>("switchOn")) {
      delta_eta_VS_et_ = book2D("delta_eta_VS_et_", ";E_{T, true} (GeV);#Delta#eta",
				ptBinsPS.size()-1, &(ptBins.front()),
				detaPS.getParameter<int32_t>("nBin"), 
				detaPS.getParameter<double>("xMin"), 
				detaPS.getParameter<double>("xMax"));
    }
    
    if (dphiPS.getParameter<bool>("switchOn")) {
      delta_phi_VS_et_ = book2D("delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
				ptBinsPS.size()-1, &(ptBins.front()),
				dphiPS.getParameter<int32_t>("nBin"), 
				dphiPS.getParameter<double>("xMin"),
				dphiPS.getParameter<double>("xMax"));
    }
    /*
    // TProfile
    if (dptOvptPS.getParameter<bool>("switchOn")) {
      profile_delta_et_Over_et_VS_et_ = bookProfile("profile_delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
						    ptBinsPS.size()-1, &(ptBins.front()), 
						    dptOvptPS.getParameter<double>("xMin"), 
						    dptOvptPS.getParameter<double>("xMax"), "" );
      profileRMS_delta_et_Over_et_VS_et_ = bookProfile("profileRMS_delta_et_Over_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
						    ptBinsPS.size()-1, &(ptBins.front()), 
						    dptOvptPS.getParameter<double>("xMin"), 
						    dptOvptPS.getParameter<double>("xMax"), "s" );
    }
    
    if (dptPS.getParameter<bool>("switchOn")) {
      profile_delta_et_VS_et_ = bookProfile("profile_delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
					    ptBinsPS.size()-1, &(ptBins.front()),
					    dptPS.getParameter<double>("xMin"), 
					    dptPS.getParameter<double>("xMax"), "" );
      profileRMS_delta_et_VS_et_ = bookProfile("profileRMS_delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}",
					    ptBinsPS.size()-1, &(ptBins.front()),
					    dptPS.getParameter<double>("xMin"), 
					    dptPS.getParameter<double>("xMax"), "s" );
    }
    
    if (detaPS.getParameter<bool>("switchOn")) {
      profile_delta_eta_VS_et_ = bookProfile("profile_delta_eta_VS_et_", ";E_{T, true} (GeV);#Delta#eta",
					     ptBinsPS.size()-1, &(ptBins.front()),
					     detaPS.getParameter<double>("xMin"), 
					     detaPS.getParameter<double>("xMax"), "" );
      profileRMS_delta_eta_VS_et_ = bookProfile("profileRMS_delta_eta_VS_et_", ";E_{T, true} (GeV);#Delta#eta",
					     ptBinsPS.size()-1, &(ptBins.front()),
					     detaPS.getParameter<double>("xMin"), 
					     detaPS.getParameter<double>("xMax"), "s" );
    }
    
    if (dphiPS.getParameter<bool>("switchOn")) {
      profile_delta_phi_VS_et_ = bookProfile("profile_delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
					     ptBinsPS.size()-1, &(ptBins.front()),
					     dphiPS.getParameter<double>("xMin"),
					     dphiPS.getParameter<double>("xMax"), "" );
      profileRMS_delta_phi_VS_et_ = bookProfile("profileRMS_delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi",
					     ptBinsPS.size()-1, &(ptBins.front()),
					     dphiPS.getParameter<double>("xMin"),
					     dphiPS.getParameter<double>("xMax"), "s" );
					     }*/

    histogramBooked_ = true;
  }
}

void MatchCandidateBenchmark::fillOne(const reco::Candidate& cand,
				      const reco::Candidate& matchedCand) {

  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;

  if (histogramBooked_) {
    if (delta_et_Over_et_VS_et_) delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
    if ( fabs(cand.eta())  <= 1.4 )
      if (BRdelta_et_Over_et_VS_et_) BRdelta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
    if ( fabs(cand.eta()) >= 1.6 && fabs(cand.eta()) <= 2.4 )
      if (ERdelta_et_Over_et_VS_et_) ERdelta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
    if (delta_et_VS_et_) delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
    if (delta_eta_VS_et_) delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
    if (delta_phi_VS_et_) delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
    /*
    // TProfile
    if (profile_delta_et_Over_et_VS_et_) {
      profile_delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
      profileRMS_delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() ); }
    if (profile_delta_et_VS_et_) {
      profile_delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
      profileRMS_delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() ); }
    if (profile_delta_eta_VS_et_) {
      profile_delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
      profileRMS_delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() ); }
    if (profile_delta_phi_VS_et_) {
      profile_delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
      profileRMS_delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() ); }
    */
  }  
}

void MatchCandidateBenchmark::fillOne(const reco::Candidate& cand,
				      const reco::Candidate& matchedCand,
				      const edm::ParameterSet& parameterSet) {
  if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) return;
  
  if (histogramBooked_) { 

    std::vector<double> ptBinsPS = parameterSet.getParameter< std::vector<double> >( "VariablePtBins" );
    edm::ParameterSet ptPS = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    std::vector<float> ptBins;
    if (ptBinsPS.size() > 1) { 
      ptBins.reserve(ptBinsPS.size());
      for (size_t i = 0; i < ptBinsPS.size(); i++) {
        ptBins.push_back(ptBinsPS[i]);
      }
    } else { 
      Int_t nFixedBins = ptPS.getParameter<int32_t>("nBin");
      ptBins.reserve(nFixedBins + 1);
      for (Int_t i = 0; i <= nFixedBins; i++) {
        ptBins.push_back( ptPS.getParameter<double>("xMin") + i*((ptPS.getParameter<double>("xMax") - ptPS.getParameter<double>("xMin")) / nFixedBins) );
      }
      ptBinsPS.resize(nFixedBins);
    }

    edm::ParameterSet dptOvptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtOvPtHistoParameter");
    if (matchedCand.pt() > ptBins.at(0)) { // underflow problem
      if (delta_et_Over_et_VS_et_) delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt() );
      if ( fabs(cand.eta()) >= dptOvptPS.getParameter<double>("BREtaMin")  &&  fabs(cand.eta()) <= dptOvptPS.getParameter<double>("BREtaMax"))
	if (BRdelta_et_Over_et_VS_et_) BRdelta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
      if ( fabs(cand.eta()) >= dptOvptPS.getParameter<double>("EREtaMin")  &&  fabs(cand.eta()) <= dptOvptPS.getParameter<double>("EREtaMax"))
	if (ERdelta_et_Over_et_VS_et_) ERdelta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
      if (delta_et_VS_et_) delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
      if (delta_eta_VS_et_) delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
      if (delta_phi_VS_et_) delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
    }
    /*
    // TProfile
    if (profile_delta_et_Over_et_VS_et_) {
      profile_delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() );
      profileRMS_delta_et_Over_et_VS_et_->Fill( matchedCand.pt(), (cand.pt() - matchedCand.pt())/matchedCand.pt() ); }
    if (profile_delta_et_VS_et_) {
      profile_delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() );
      profileRMS_delta_et_VS_et_->Fill( matchedCand.pt(), cand.pt() - matchedCand.pt() ); }
    if (profile_delta_eta_VS_et_) {
      profile_delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() );
      profileRMS_delta_eta_VS_et_->Fill( matchedCand.pt(), cand.eta() - matchedCand.eta() ); }
    if (profile_delta_phi_VS_et_) {
      profile_delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() );
      profileRMS_delta_phi_VS_et_->Fill( matchedCand.pt(), cand.phi() - matchedCand.phi() ); }
    */

    for (size_t i = 0; i < pTRes_.size(); i++) {
      if (matchedCand.pt() >= ptBins.at(i) && matchedCand.pt() < ptBins.at(i+1)) {
	if (pTRes_[i]) pTRes_[i]->Fill( (cand.pt() - matchedCand.pt()) / matchedCand.pt() ) ;
      	if ( fabs(cand.eta()) >= dptOvptPS.getParameter<double>("BREtaMin")  &&  fabs(cand.eta()) <= dptOvptPS.getParameter<double>("BREtaMax"))
	  if (BRpTRes_[i]) BRpTRes_[i]->Fill( (cand.pt() - matchedCand.pt()) / matchedCand.pt() ) ; // Fill Barrel
      	if ( fabs(cand.eta()) >= dptOvptPS.getParameter<double>("EREtaMin")  &&  fabs(cand.eta()) <= dptOvptPS.getParameter<double>("EREtaMax"))
	  if (ERpTRes_[i]) ERpTRes_[i]->Fill( (cand.pt() - matchedCand.pt()) / matchedCand.pt() ) ; // Fill Endcap
      }
    }
  }
}

