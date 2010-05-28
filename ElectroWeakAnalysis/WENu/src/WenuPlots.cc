// -*- C++ -*-
//
// Package:    WenuPlots
// Class:      WenuPlots
// 
/*

 Description: 
    this is an analyzer that reads pat::CompositeCandidate WenuCandidates
    and creates some plots
 Implementation:
    The code takes the output of the WenuCandidateFilter and
    * implements on them  a user defined selection 
    * implements the selection with one cut (configurable which cut) inverted
    * creates a set of basic plots with the Wenu Candidate distribution
      vs MET, MT etc. These plots are stored in a root file
    If you have several root files from different runs you have to run a macro
    to combine the output and have the final plots

    This analyser is PAT based in the sense that it reads CompositeCandidates,
    which are composed of a pat::MET plus a pat::Electron. You normally
    don't have to change this file when the CMSSW version changes because it
    contains only methods from the stable core of pat Objects. Most
    version dependent changes should be in WenuCandidateFilter.cc
 TO DO LIST:
    * more plots to be added
    * there should be an base Plots class from which WenuPlots and ZeePlots
      inherit. this makes sense since they have so many common methods

  Changes Log:
  12Feb09  First Release of the code for CMSSW_2_2_X
  16Sep09  tested that it works with 3_1_2 as well 
  09Sep09  added one extra iso with the name userIso_XX_
  23Feb09  added option to include extra IDs that are in CMSSW, such as
           categorized, likehood etc
           added extra variables TIP and E/P  
  27May10  changes to apply the Spring10 selections, relative isolations
           the 3 default ones, pat user isolations added in the end
           change to framework independent variable definitions 
	   double->Double_t etc and math.h functions from TMath
  Contact: 
  Nikolaos Rompotis  -  Nikolaos.Rompotis@Cern.ch
  Imperial College London


*/
//
// Original Author:  Nikolaos Rompotis


#include "ElectroWeakAnalysis/WENu/interface/WenuPlots.h"

WenuPlots::WenuPlots(const edm::ParameterSet& iConfig)

{
////////////////////////////////////////////////////////////////////////////
//                   I N P U T      P A R A M E T E R S
////////////////////////////////////////////////////////////////////////////
//
///////
//  WENU COLLECTION   //////////////////////////////////////////////////////
//
  
  wenuCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>
    ("wenuCollectionTag");
  //
  // code parameters
  //
  std::string outputFile_D = "histos.root";
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile",
							   outputFile_D);
  //
  // use of precalculatedID
  // if you use it, then no other cuts are applied
  usePrecalcID_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID",false);
  if (usePrecalcID_) {
    usePrecalcIDType_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType");    
    usePrecalcIDSign_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign","=");    
    usePrecalcIDValue_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue");
  }
  useValidFirstPXBHit_ = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit",false);
  useConversionRejection_ = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection",false);
  useExpectedMissingHits_ = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits",false);
  maxNumberOfExpectedMissingHits_ = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits",1);
  if (useValidFirstPXBHit_) std::cout << "WenuPlots: Warning: you have demanded a valid 1st layer PXB hit" << std::endl;
  if (useConversionRejection_) std::cout << "WenuPlots: Warning: you have demanded egamma conversion rejection criteria to be applied" << std::endl;
  if (useExpectedMissingHits_) std::cout << "WenuPlots: Warning: you have demanded at most " 
      <<maxNumberOfExpectedMissingHits_ << " missing inner hits "<< std::endl;
  if (useValidFirstPXBHit_ || useExpectedMissingHits_ || useConversionRejection_) {
    usePreselection_ = true;
  } else { usePreselection_ = false; }
  //
  // the selection cuts:
  trackIso_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EB", 1000.);
  ecalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EB", 1000.);
  hcalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EB", 1000.);
  //
  trackIso_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EE", 1000.);
  ecalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EE", 1000.);
  hcalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EE", 1000.);
  //
  sihih_EB_ = iConfig.getUntrackedParameter<Double_t>("sihih_EB");
  dphi_EB_ = iConfig.getUntrackedParameter<Double_t>("dphi_EB");
  deta_EB_ = iConfig.getUntrackedParameter<Double_t>("deta_EB");
  hoe_EB_ = iConfig.getUntrackedParameter<Double_t>("hoe_EB");
  cIso_EB_ = iConfig.getUntrackedParameter<Double_t>("cIso_EB", 1000.);
  tip_bspot_EB_=iConfig.getUntrackedParameter<Double_t>("tip_bspot_EB", 1000.);
  eop_EB_=iConfig.getUntrackedParameter<Double_t>("eop_EB", 1000.);
  //
  sihih_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih_EE");
  dphi_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi_EE");
  deta_EE_ = iConfig.getUntrackedParameter<Double_t>("deta_EE");
  hoe_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe_EE");
  cIso_EE_ = iConfig.getUntrackedParameter<Double_t>("cIso_EE", 1000.);
  tip_bspot_EE_=iConfig.getUntrackedParameter<Double_t>("tip_bspot_EE", 1000.);
  eop_EE_=iConfig.getUntrackedParameter<Double_t>("eop_EE", 1000.);
  //
  trackIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIsoUser_EB", 1000.);
  ecalIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIsoUser_EB", 1000.);
  hcalIsoUser_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIsoUser_EB", 1000.);
  trackIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIsoUser_EE", 1000.);
  ecalIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIsoUser_EE", 1000.);
  hcalIsoUser_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIsoUser_EE", 1000.);
  //
  trackIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EB_inv", false);
  ecalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EB_inv", false);
  hcalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EB_inv", false);
  //
  trackIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EE_inv", false);
  ecalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EE_inv", false);
  hcalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EE_inv", false);
  //
  sihih_EB_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EB_inv", false);
  dphi_EB_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EB_inv", false);
  deta_EB_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EB_inv", false);
  hoe_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EB_inv", false);
  cIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("cIso_EB_inv", false);
  tip_bspot_EB_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EB_inv", false);
  eop_EB_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EB_inv", false);
  //
  sihih_EE_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EE_inv", false);
  dphi_EE_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EE_inv", false);
  deta_EE_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EE_inv", false);
  hoe_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EE_inv", false);
  cIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("cIso_EE_inv", false);
  tip_bspot_EE_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EE_inv", false);
  eop_EE_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EE_inv", false);
  //
  trackIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EB_inv", false);
  ecalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EB_inv", false);
  hcalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EB_inv", false);
  trackIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EE_inv", false);
  ecalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EE_inv", false);
  hcalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EE_inv", false);

}



WenuPlots::~WenuPlots()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WenuPlots::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
  using namespace std;
  //
  //  Get the collections here
  //
  edm::Handle<pat::CompositeCandidateCollection> WenuCands;
  iEvent.getByLabel(wenuCollectionTag_, WenuCands);

  if (not WenuCands.isValid()) {
    cout << "Warning: no wenu candidates in this event..." << endl;
    return;
  }
  // calculate the beam spot position for the TIP calculation
  //
  // this is how one should do it, however, the beam spot is given in
  // the pat electron producer cfi, hence it is stored there
  // The reason that we don't do like that is that we want if possible the
  // code to run only on WenuCandidate objects and nothing else
  // facilitating the analysis from super-skimmed edmFiles
  //
  // edm::Handle<reco::BeamSpot> pBeamSpot;
  // if(iEvent.getByLabel("offlineBeamSpot", pBeamSpot)) {
  //   const reco::BeamSpot *bspot = pBeamSpot.product();
  //   bspotPosition_ = bspot->position();
  // } else {
  //   std::cout << "Offline beam spot was not found with collection name " 
  // 	      << " offlineBeamSpot  --> it will be set to zero" << std::endl;
  //   bspotPosition_.SetXYZ(0,0,0);
  // }
  //
  const pat::CompositeCandidateCollection *wcands = WenuCands.product();
  const pat::CompositeCandidateCollection::const_iterator 
    wenuIter = wcands->begin();
  const pat::CompositeCandidate wenu = *wenuIter;
  //
  // get the parts of the composite candidate:
  const pat::Electron * myElec=
    dynamic_cast<const pat::Electron*> (wenu.daughter("electron"));
  const pat::MET * myMet=
    dynamic_cast<const pat::MET*> (wenu.daughter("met"));
  //
  // if you want some preselection: Conv rejection, hit pattern 
  if (usePreselection_) {
    if (not PassPreselectionCriteria(myElec)) return;
  }
  //
  // some variables here
  Double_t scEta = myElec->superCluster()->eta();
  Double_t scPhi = myElec->superCluster()->phi();
  Double_t scEt = myElec->superCluster()->energy()/TMath::CosH(scEta);
  Double_t met    = myMet->et();
  Double_t metPhi = myMet->phi();
  Double_t mt  = TMath::Sqrt(2.0*scEt*met*(1.0-(TMath::Cos(scPhi)*TMath::Cos(metPhi)+TMath::Sin(scPhi)*TMath::Sin(metPhi))));

  Double_t trackIso = myElec->userIsolation(pat::TrackIso);
  Double_t ecalIso = myElec->userIsolation(pat::EcalIso);
  Double_t hcalIso = myElec->userIsolation(pat::HcalIso);
  Double_t sihih = myElec->scSigmaIEtaIEta();
  Double_t dphi = myElec->deltaPhiSuperClusterTrackAtVtx();
  Double_t deta = myElec->deltaEtaSuperClusterTrackAtVtx();
  Double_t HoE = myElec->hadronicOverEm();
  //
  //
  //
  // the inverted selection plots:
  // only if not using precalcID
  if (not usePrecalcID_) {
    if (CheckCutsInverse(myElec)){
      //std::cout << "-----------------INVERSION-----------passed" << std::endl;
      h_met_inverse->Fill(met);
      h_mt_inverse->Fill(mt);
      if(TMath::Abs(scEta)<1.479){
	h_met_inverse_EB->Fill(met);
	h_mt_inverse_EB->Fill(mt);
      }
      if(TMath::Abs(scEta)>1.479){
	h_met_inverse_EE->Fill(met);
	h_mt_inverse_EE->Fill(mt);
      }
    }
  }
  //
  ///////////////////////////////////////////////////////////////////////
  //
  // N-1 plots: plot some variable so that all the other cuts are satisfied
  //
  // make these plots only if you have the normal selection, not pre-calced
  if (not usePrecalcID_) {
    if ( TMath::Abs(scEta) < 1.479) { // reminder: the precise fiducial cuts are in
      // in the filter
      if (CheckCutsNminusOne(myElec, 0)) 
	h_trackIso_eb_NmOne->Fill(trackIso);
    }
    else {
      if (CheckCutsNminusOne(myElec, 0)) 
	h_trackIso_ee_NmOne->Fill(trackIso);
    }
  }
  //
  // SELECTION APPLICATION
  //
  // from here on you have only events that pass the full selection
  if (not CheckCuts(myElec)) return;
  //////////////////////////////////////////////////////////////////////

  h_met->Fill(met);
  h_mt->Fill(mt);
  if(TMath::Abs(scEta)<1.479){
    h_met_EB->Fill(met);
    h_mt_EB->Fill(mt);

    h_EB_trkiso->Fill( trackIso );
    h_EB_ecaliso->Fill( ecalIso );
    h_EB_hcaliso->Fill( hcalIso );
    h_EB_sIetaIeta->Fill( sihih );
    h_EB_dphi->Fill( dphi );
    h_EB_deta->Fill( deta );
    h_EB_HoE->Fill( HoE );

  }
  if(TMath::Abs(scEta)>1.479){
    h_met_EE->Fill(met);
    h_mt_EE->Fill(mt);

    h_EE_trkiso->Fill( trackIso );
    h_EE_ecaliso->Fill( ecalIso );
    h_EE_hcaliso->Fill( hcalIso );
    h_EE_sIetaIeta->Fill( sihih );
    h_EE_dphi->Fill( dphi );
    h_EE_deta->Fill( deta );
    h_EE_HoE->Fill( HoE );

  }
  // uncomment for debugging purposes
  /*
  std::cout << "tracIso: " <<  trackIso << ", " << myElec->trackIso() << ", ecaliso: " << ecalIso 
	    << ", " << myElec->ecalIso() << ", hcaliso: " << hcalIso << ", "  << myElec->hcalIso() 
	    << ", mishits: " 
	    << myElec->gsfTrack()->trackerExpectedHitsInner().numberOfHits()
	    << std::endl;
  std::cout << "Electron ID: 95relIso=" << myElec->electronID("simpleEleId95relIso")  
	    << " 90relIso=" << myElec->electronID("simpleEleId90relIso") 
	    << " 85relIso=" << myElec->electronID("simpleEleId85relIso") 
	    << " 80relIso=" << myElec->electronID("simpleEleId80relIso") 
	    << " 70relIso=" << myElec->electronID("simpleEleId70relIso") 
	    << " 60relIso=" << myElec->electronID("simpleEleId60relIso") 
	    << " 95cIso=" << myElec->electronID("simpleEleId95cIso") 
	    << " 90cIso=" << myElec->electronID("simpleEleId90cIso") 
	    << " 85cIso=" << myElec->electronID("simpleEleId85cIso") 
	    << " 80cIso=" << myElec->electronID("simpleEleId80cIso") 
	    << " 70cIso=" << myElec->electronID("simpleEleId70cIso") 
	    << " 60cIso=" << myElec->electronID("simpleEleId60cIso") 
	    << std::endl;
  std::cout << "mySelection: " << (CheckCuts(myElec) && PassPreselectionCriteria(myElec)) << endl;
  */
  h_scEt->Fill(scEt);
  h_scEta->Fill(scEta);
  h_scPhi->Fill(scPhi);

}

/***********************************************************************
 *
 *  Checking Cuts and making selections:
 *  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *  all the available methods take input a pointer to a  pat::Electron
 *
 *  Bool_t  CheckCuts(const pat::Electron *): 
 *                            true if the input selection is satisfied
 *  Bool_t  CheckCutsInverse(const pat::Electron *ele):
 *            true if the cuts with inverted the ones specified in the
 *            cfg are satisfied
 *  Bool_t  CheckCutsNminusOne(const pat::Electron *ele, int jj):
 *             true if all the cuts with cut #jj ignored are satisfied
 *
 ***********************************************************************/
Bool_t WenuPlots::CheckCuts( const pat::Electron *ele)
{
  if (usePrecalcID_) {
    if (not ele-> isElectronIDAvailable(usePrecalcIDType_)) {
      std::cout << "Error! not existing ID with name: "
		<< usePrecalcIDType_ << " function will return true!"
		<< std::endl;
      return true;
    }
    Double_t val = ele->electronID(usePrecalcIDType_);
    if (usePrecalcIDSign_ == "<") {
      return val < usePrecalcIDValue_;
    }
    else if (usePrecalcIDSign_ == ">") {
      return val > usePrecalcIDValue_;
    }
    else { // equality: it returns 0,1,2,3 but as float
      return TMath::Abs(val-usePrecalcIDValue_)<0.1;
    }
  } 
  else {
    for (int i=0; i<nBarrelVars_; ++i) {
      if (not CheckCut(ele, i)) return false;
    }
    return true;
  }
}
/////////////////////////////////////////////////////////////////////////

Bool_t WenuPlots::CheckCutsInverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCutInv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCutsNminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCut(const pat::Electron *ele, int i) {
  Double_t fabseta = TMath::Abs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::CheckCutInv(const pat::Electron *ele, int i) {
  Double_t fabseta = TMath::Abs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    if (InvVars_[i]) 
    return TMath::Abs(ReturnCandVar(ele, i))>CutVars_[i];
    return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  if (InvVars_[i+nBarrelVars_]) {
    if (InvVars_[i])
      return TMath::Abs(ReturnCandVar(ele, i))>CutVars_[i+nBarrelVars_];    
  }
  return TMath::Abs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
////////////////////////////////////////////////////////////////////////
Double_t WenuPlots::ReturnCandVar(const pat::Electron *ele, int i) {
  if (i==0) return ele->dr03TkSumPt()/ele->p4().Pt();
  else if (i==1) return ele->dr03EcalRecHitSumEt()/ele->p4().Pt();
  else if (i==2) return ele->dr03HcalTowerSumEt()/ele->p4().Pt();
  else if (i==3) return ele->scSigmaIEtaIEta();
  else if (i==4) return ele->deltaPhiSuperClusterTrackAtVtx();
  else if (i==5) return ele->deltaEtaSuperClusterTrackAtVtx();
  else if (i==6) return ele->hadronicOverEm();
  else if (i==7) {
    if (ele->isEB()){
      return ( ele->dr03TkSumPt()+std::max(float(0.),ele->dr03EcalRecHitSumEt()-1)
	       + ele->dr03HcalTowerSumEt())/ele->p4().Pt(); }
    else { // pedestal subtraction is only in barrel
      return ( ele->dr03TkSumPt()+ele->dr03EcalRecHitSumEt()
	       + ele->dr03HcalTowerSumEt())/ele->p4().Pt(); }
  }
  //  else if (i==8) return ele->gsfTrack()->dxy(bspotPosition_);
  else if (i==8) return ele->dB();
  else if (i==9) return ele->eSuperClusterOverP();
  else if (i==10) return ele->userIsolation(pat::TrackIso);
  else if (i==11) return ele->userIsolation(pat::EcalIso);
  else if (i==12) return ele->userIsolation(pat::HcalIso);  
  std::cout << "Error in WenuPlots::ReturnCandVar" << std::endl;
  return -1.;

}
/////////////////////////////////////////////////////////////////////////
Bool_t WenuPlots::PassPreselectionCriteria(const pat::Electron *ele) {
  Bool_t passConvRej = true;
  Bool_t passPXB = true;
  Bool_t passEMH = true;
  if (useConversionRejection_) {
    if (ele->hasUserInt("PassConversionRejection")) {
      //std::cout << "con rej: " << ele->userInt("PassConversionRejection") << std::endl;
      if (not (ele->userInt("PassConversionRejection")==1)) passConvRej = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Conversion Rejection Request Disregarded: "
		<< "you must calculate it before " << std::endl;
      // return true;
    }
  }
  if (useValidFirstPXBHit_) {
    if (ele->hasUserInt("PassValidFirstPXBHit")) {
      //std::cout << "valid1stPXB: " << ele->userInt("PassValidFirstPXBHit") << std::endl;
      if (not (ele->userInt("PassValidFirstPXBHit")==1)) passPXB = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Valid First PXB Hit Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true;
    }
  }
  if (useExpectedMissingHits_) {
    if (ele->hasUserInt("NumberOfExpectedMissingHits")) {
      //std::cout << "missing hits: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;
      if (ele->userInt("NumberOfExpectedMissingHits")>maxNumberOfExpectedMissingHits_) 
	passEMH = false;
    }
    else {
      std::cout << "WenuPlots: WARNING: Number of Expected Missing Hits Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true;
    }
  }
  return passConvRej && passPXB && passEMH;
}
// ------------ method called once each job just before starting event loop  --
void 
WenuPlots::beginJob()
{
  //std::cout << "In beginJob()" << std::endl;
  //  Double_t Pi = TMath::Pi();
  //  TString histo_file = outputFile_;
  //  histofile = new TFile( histo_file,"RECREATE");

  h_met         = new TH1F("h_met",         "h_met",         200, 0, 200);
  h_met_inverse = new TH1F("h_met_inverse", "h_met_inverse", 200, 0, 200);

  h_mt         = new TH1F("h_mt",         "h_mt",         200, 0, 200);
  h_mt_inverse = new TH1F("h_mt_inverse", "h_mt_inverse", 200, 0, 200);


  h_met_EB         = new TH1F("h_met_EB",         "h_met_EB",         200, 0, 200);
  h_met_inverse_EB = new TH1F("h_met_inverse_EB", "h_met_inverse_EB", 200, 0, 200);

  h_mt_EB         = new TH1F("h_mt_EB",         "h_mt_EB",         200, 0, 200);
  h_mt_inverse_EB = new TH1F("h_mt_inverse_EB", "h_mt_inverse_EB", 200, 0, 200);


  h_met_EE         = new TH1F("h_met_EE",         "h_met_EE",         200, 0, 200);
  h_met_inverse_EE = new TH1F("h_met_inverse_EE", "h_met_inverse_EE", 200, 0, 200);

  h_mt_EE         = new TH1F("h_mt_EE",         "h_mt_EE",         200, 0, 200);
  h_mt_inverse_EE = new TH1F("h_mt_inverse_EE", "h_mt_inverse_EE", 200, 0, 200);


  h_scEt  = new TH1F("h_scEt",  "h_scEt",  200,  0, 100);
  h_scEta = new TH1F("h_scEta", "h_scEta", 200, -3, 3);
  h_scPhi = new TH1F("h_scPhi", "h_scPhi", 200, -4, 4);


  //VALIDATION PLOTS
  //EB
  h_EB_trkiso = new TH1F("h_EB_trkiso","h_EB_trkiso",200 , 0.0, 9.0);
  h_EB_ecaliso = new TH1F("h_EB_ecaliso","h_EB_ecaliso",200, 0.0 , 9.0);
  h_EB_hcaliso = new TH1F("h_EB_hcaliso","h_EB_hcaliso",200, 0.0 , 9.0);
  h_EB_sIetaIeta = new TH1F("h_EB_sIetaIeta","h_EB_sIetaIeta",200, 0.0 , 0.02 );
  h_EB_dphi = new TH1F("h_EB_dphi","h_EB_dphi",200, -0.03 , 0.03 );
  h_EB_deta = new TH1F("h_EB_deta","h_EB_deta",200, -0.01 , 0.01) ;
  h_EB_HoE = new TH1F("h_EB_HoE","h_EB_HoE",200, 0.0 , 0.2 );
  //EE
  h_EE_trkiso = new TH1F("h_EE_trkiso","h_EE_trkiso",200 , 0.0, 9.0);
  h_EE_ecaliso = new TH1F("h_EE_ecaliso","h_EE_ecaliso",200, 0.0 , 9.0);
  h_EE_hcaliso = new TH1F("h_EE_hcaliso","h_EE_hcaliso",200, 0.0 , 9.0);
  h_EE_sIetaIeta = new TH1F("h_EE_sIetaIeta","h_EE_sIetaIeta",200, 0.0 , 0.1 );
  h_EE_dphi = new TH1F("h_EE_dphi","h_EE_dphi",200, -0.03 , 0.03 );
  h_EE_deta = new TH1F("h_EE_deta","h_EE_deta",200, -0.01 , 0.01) ;
  h_EE_HoE = new TH1F("h_EE_HoE","h_EE_HoE",200, 0.0 , 0.2 );


  //
  //
  h_trackIso_eb_NmOne = 
    new TH1F("h_trackIso_eb_NmOne","trackIso EB N-1 plot",80,0,8);
  h_trackIso_ee_NmOne = 
    new TH1F("h_trackIso_ee_NmOne","trackIso EE N-1 plot",80,0,8);

  
  // if you add some new variable change the nBarrelVars_ accordingly
  // reminder: in the current implementation you must have the same number
  //  of vars in both barrel and endcaps
  nBarrelVars_ = 13;
  //
  // Put EB variables together and EE variables together
  // number of barrel variables = number of endcap variable
  // if you don't want to use some variable put a very high cut
  CutVars_.push_back( trackIso_EB_ );//0
  CutVars_.push_back( ecalIso_EB_ ); //1
  CutVars_.push_back( hcalIso_EB_ ); //2
  CutVars_.push_back( sihih_EB_ );   //3
  CutVars_.push_back( dphi_EB_ );    //4
  CutVars_.push_back( deta_EB_ );    //5
  CutVars_.push_back( hoe_EB_ );     //6
  CutVars_.push_back( cIso_EB_ );    //7
  CutVars_.push_back( tip_bspot_EB_);//8
  CutVars_.push_back( eop_EB_ );     //9
  CutVars_.push_back( trackIsoUser_EB_ );//10
  CutVars_.push_back( ecalIsoUser_EB_  );//11
  CutVars_.push_back( hcalIsoUser_EB_  );//12
  //
  CutVars_.push_back( trackIso_EE_);//0
  CutVars_.push_back( ecalIso_EE_); //1
  CutVars_.push_back( hcalIso_EE_); //2
  CutVars_.push_back( sihih_EE_);   //3
  CutVars_.push_back( dphi_EE_);    //4
  CutVars_.push_back( deta_EE_);    //5
  CutVars_.push_back( hoe_EE_ );    //6 
  CutVars_.push_back( cIso_EE_ );   //7 
  CutVars_.push_back(tip_bspot_EE_);//8
  CutVars_.push_back( eop_EE_ );    //9
  CutVars_.push_back( trackIsoUser_EE_ );//10
  CutVars_.push_back( ecalIsoUser_EE_  );//11
  CutVars_.push_back( hcalIsoUser_EE_  );//12
  //
  InvVars_.push_back( trackIso_EB_inv);//0
  InvVars_.push_back( ecalIso_EB_inv); //1
  InvVars_.push_back( hcalIso_EB_inv); //2
  InvVars_.push_back( sihih_EB_inv);   //3
  InvVars_.push_back( dphi_EB_inv);    //4
  InvVars_.push_back( deta_EB_inv);    //5
  InvVars_.push_back( hoe_EB_inv);     //6
  InvVars_.push_back( cIso_EB_inv);    //7
  InvVars_.push_back(tip_bspot_EB_inv);//8
  InvVars_.push_back( eop_EB_inv);     //9
  InvVars_.push_back( trackIsoUser_EB_inv );//10
  InvVars_.push_back( ecalIsoUser_EB_inv  );//11
  InvVars_.push_back( hcalIsoUser_EB_inv  );//12
  //
  InvVars_.push_back( trackIso_EE_inv);//0
  InvVars_.push_back( ecalIso_EE_inv); //1
  InvVars_.push_back( hcalIso_EE_inv); //2
  InvVars_.push_back( sihih_EE_inv);   //3
  InvVars_.push_back( dphi_EE_inv);    //4
  InvVars_.push_back( deta_EE_inv);    //5
  InvVars_.push_back( hoe_EE_inv);     //6
  InvVars_.push_back( cIso_EE_inv);    //7
  InvVars_.push_back(tip_bspot_EE_inv);//8
  InvVars_.push_back( eop_EE_inv);     //9
  InvVars_.push_back( trackIsoUser_EE_inv );//10
  InvVars_.push_back( ecalIsoUser_EE_inv  );//11
  InvVars_.push_back( hcalIsoUser_EE_inv  );//12
  //


}

// ------------ method called once each job just after ending the event loop  -
void 
WenuPlots::endJob() {
  TFile * newfile = new TFile(TString(outputFile_),"RECREATE");
  //
  // for consistency all the plots are in the root file
  // even though they may be empty (in the case when
  // usePrecalcID_== true inverted and N-1 are empty)
  h_met->Write();
  h_met_inverse->Write();
  h_mt->Write();
  h_mt_inverse->Write();

  h_met_EB->Write();
  h_met_inverse_EB->Write();
  h_mt_EB->Write();
  h_mt_inverse_EB->Write();

  h_met_EE->Write();
  h_met_inverse_EE->Write();
  h_mt_EE->Write();
  h_mt_inverse_EE->Write();

  h_scEt->Write();
  h_scEta->Write();
  h_scPhi->Write();

  h_EB_trkiso->Write();
  h_EB_ecaliso->Write();
  h_EB_hcaliso->Write();
  h_EB_sIetaIeta->Write();
  h_EB_dphi->Write();
  h_EB_deta->Write();
  h_EB_HoE->Write();

  h_EE_trkiso->Write();
  h_EE_ecaliso->Write();
  h_EE_hcaliso->Write();
  h_EE_sIetaIeta->Write();
  h_EE_dphi->Write();
  h_EE_deta->Write();
  h_EE_HoE->Write();

  //
  h_trackIso_eb_NmOne->Write();
  h_trackIso_ee_NmOne->Write();
  //
  newfile->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(WenuPlots);
