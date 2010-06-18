// -*- C++ -*-
//
// Package:    ZeePlots
// Class:      ZeePlots
// 
/*

 Description: <one line class summary>
    this is an analyzer that reads pat::CompositeCandidate ZeeCandidates
    and creates some plots
    For more details see also WenuPlots class description
 Implementation:
  09Dec09: option to have a different selection for the 2nd leg of the Z added
  24Feb10: more variables added E/P and TIP
           option to choose CMSSW defined electron ID, the same or different
           for each leg
  18June10: bug corrected thanks to Jie Chen
*/
//
// Original Author:  Nikolaos Rompotis


#include "ElectroWeakAnalysis/ZEE/interface/ZeePlots.h"

ZeePlots::ZeePlots(const edm::ParameterSet& iConfig)

{
////////////////////////////////////////////////////////////////////////////
//                   I N P U T      P A R A M E T E R S
////////////////////////////////////////////////////////////////////////////
//
///////
//  ZEE COLLECTION   //////////////////////////////////////////////////////
//
  
  zeeCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>
    ("zeeCollectionTag");
  //
  // code parameters
  //
  std::string outputFile_D = "histos.root";
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile",
							   outputFile_D);
  //
  //
  // use of precalculatedID
  // if you use it, then no other cuts are applied
  usePrecalcID_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID",false);
  if (usePrecalcID_) {
    usePrecalcIDType_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDType");    
    usePrecalcIDSign_ = iConfig.getUntrackedParameter<std::string>("usePrecalcIDSign","=");    
    usePrecalcIDValue_= iConfig.getUntrackedParameter<Double_t>("usePrecalcIDValue");
    std::cout << "ZeePlots: WARNING: you have chosen to use CMSSW"
	      << " precalculated ID with name: >>> " << usePrecalcIDType2_
	      << " <<< such that the value map " << usePrecalcIDSign2_ << " "
	      << usePrecalcIDValue2_ << std::endl;
  }
  useValidFirstPXBHit_ = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit",false);
  useConversionRejection_ = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection",false);
  useExpectedMissingHits_ = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits",false);
  useValidFirstPXBHit2_ = iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit2",false);
  useConversionRejection2_ = iConfig.getUntrackedParameter<Bool_t>("useConversionRejection2",false);
  useExpectedMissingHits2_ = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits2",false);
  maxNumberOfExpectedMissingHits_ = iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits",1);
  maxNumberOfExpectedMissingHits2_= iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits2",1);
  if (useValidFirstPXBHit_ || useConversionRejection_ || useExpectedMissingHits_ || useValidFirstPXBHit2_
      || useConversionRejection2_ || useExpectedMissingHits2_) usePreselection_ = true;
  else usePreselection_=false;
  if (useValidFirstPXBHit_) 
    std::cout << "ZeePlots: Warning: you have demanded ValidFirstPXBHit in 1st electron" << std::endl;
  if (useValidFirstPXBHit2_) 
    std::cout << "ZeePlots: Warning: you have demanded ValidFirstPXBHit in 2nd electron" << std::endl;
  if (useConversionRejection_)
    std::cout << "ZeePlots: Warning: you have demanded Conversion Rejection in 1st electron" << std::endl;
  if (useConversionRejection2_)
    std::cout << "ZeePlots: Warning: you have demanded Conversion Rejection in 2nd electron" << std::endl;
  if (useExpectedMissingHits_)
    std::cout << "ZeePlots: Warning: you have demanded Expected Missing Hits in 1st electron " 
	      << "no more than " << maxNumberOfExpectedMissingHits_    << std::endl;
  if (useExpectedMissingHits2_)
    std::cout << "ZeePlots: Warning: you have demanded Expected Missing Hits in 2nd electron " 
	      << "no more than " << maxNumberOfExpectedMissingHits2_    << std::endl;
  //
  //
  // the selection cuts:
  trackIso_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EB");
  ecalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EB");
  hcalIso_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EB");
  //
  trackIso_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIso_EE");
  ecalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIso_EE");
  hcalIso_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIso_EE");
  //
  sihih_EB_ = iConfig.getUntrackedParameter<Double_t>("sihih_EB");
  dphi_EB_ = iConfig.getUntrackedParameter<Double_t>("dphi_EB");
  deta_EB_ = iConfig.getUntrackedParameter<Double_t>("deta_EB");
  hoe_EB_ = iConfig.getUntrackedParameter<Double_t>("hoe_EB");
  cIso_EB_ = iConfig.getUntrackedParameter<Double_t>("cIso_EB",1000.);
  tip_bspot_EB_=iConfig.getUntrackedParameter<Double_t>("tip_bspot_EB", 1000.);
  eop_EB_=iConfig.getUntrackedParameter<Double_t>("eop_EB", 1000.);
  //
  sihih_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih_EE");
  dphi_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi_EE");
  deta_EE_ = iConfig.getUntrackedParameter<Double_t>("deta_EE");
  hoe_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe_EE");
  cIso_EE_ = iConfig.getUntrackedParameter<Double_t>("cIso_EE",1000.);
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
  trackIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EB_inv", 
							    false);
  ecalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EB_inv",
							   false);
  hcalIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EB_inv",
							   false);
  //
  trackIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIso_EE_inv",
							    false);
  ecalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIso_EE_inv",
							   false);
  hcalIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIso_EE_inv",
							   false);
  //
  sihih_EB_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EB_inv", false);
  dphi_EB_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EB_inv",false);
  deta_EB_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EB_inv",false);
  hoe_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EB_inv",false);
  cIso_EB_inv=iConfig.getUntrackedParameter<Bool_t>("cIso_EB_inv",false);
  tip_bspot_EB_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EB_inv", 
							 false);
  eop_EB_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EB_inv", false);
  //
  sihih_EE_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EE_inv", false);
  dphi_EE_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EE_inv", false);
  deta_EE_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EE_inv",false);
  hoe_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EE_inv",false);
  cIso_EE_inv=iConfig.getUntrackedParameter<Bool_t>("cIso_EE_inv",false);
  tip_bspot_EE_inv=iConfig.getUntrackedParameter<Bool_t>("tip_bspot_EE_inv", 
							 false);
  eop_EE_inv=iConfig.getUntrackedParameter<Bool_t>("eop_EE_inv", false);
  //
  trackIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EB_inv", false);
  ecalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EB_inv", false);
  hcalIsoUser_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EB_inv", false);
  trackIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("trackIsoUser_EE_inv", false);
  ecalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("ecalIsoUser_EE_inv", false);
  hcalIsoUser_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hcalIsoUser_EE_inv", false);
  //
  useDifferentSecondLegSelection_ = iConfig.getUntrackedParameter<Bool_t>
    ("useDifferentSecondLegSelection",false);
  if (useDifferentSecondLegSelection_) {
    std::cout << "ZeePlots: WARNING: you have chosen to use a different "
	      << " selection for the 2nd leg of the Z" << std::endl;
    trackIso2_EB_ = iConfig.getUntrackedParameter<Double_t>("trackIso2_EB");
    ecalIso2_EB_ = iConfig.getUntrackedParameter<Double_t>("ecalIso2_EB");
    hcalIso2_EB_ = iConfig.getUntrackedParameter<Double_t>("hcalIso2_EB");
    //
    trackIso2_EE_ = iConfig.getUntrackedParameter<Double_t>("trackIso2_EE");
    ecalIso2_EE_ = iConfig.getUntrackedParameter<Double_t>("ecalIso2_EE");
    hcalIso2_EE_ = iConfig.getUntrackedParameter<Double_t>("hcalIso2_EE");
    //
    sihih2_EB_ = iConfig.getUntrackedParameter<Double_t>("sihih2_EB");
    dphi2_EB_ = iConfig.getUntrackedParameter<Double_t>("dphi2_EB");
    deta2_EB_ = iConfig.getUntrackedParameter<Double_t>("deta2_EB");
    hoe2_EB_ = iConfig.getUntrackedParameter<Double_t>("hoe2_EB");
    cIso2_EB_=iConfig.getUntrackedParameter<Double_t>("cIso2_EB", 1000.);
    tip_bspot2_EB_=iConfig.getUntrackedParameter<Double_t>("tip_bspot2_EB", 
							   1000.);
    eop2_EB_=iConfig.getUntrackedParameter<Double_t>("eop2_EB", 1000.);
    //
    sihih2_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih2_EE");
    dphi2_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi2_EE");
    deta2_EE_ = iConfig.getUntrackedParameter<Double_t>("deta2_EE");
    hoe2_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe2_EE");
    cIso2_EE_=iConfig.getUntrackedParameter<Double_t>("cIso2_EE", 1000.);
    tip_bspot2_EE_=iConfig.getUntrackedParameter<Double_t>("tip_bspot2_EE", 
							   1000.);
    eop2_EE_=iConfig.getUntrackedParameter<Double_t>("eop2_EE", 1000.);

    usePrecalcID2_ = iConfig.getUntrackedParameter<Bool_t>("usePrecalcID2",
							   false);
    if (usePrecalcID2_) {
      usePrecalcIDType2_ = iConfig.getUntrackedParameter<std::string>
	("usePrecalcIDType2");    
      usePrecalcIDSign2_ = iConfig.getUntrackedParameter<std::string>
	("usePrecalcIDSign2","=");    
      usePrecalcIDValue2_= iConfig.getUntrackedParameter<Double_t>
	("usePrecalcIDValue2");
      std::cout << "ZeePlots: WARNING: 2nd leg ID:you have chosen to use CMSSW"
		<< " precalculated ID with name: >>> " << usePrecalcIDType_
		<< " <<< such that the value map " << usePrecalcIDSign_ << " "
		<< usePrecalcIDValue_ << std::endl;
    }
  }
  else {
    trackIso2_EB_ = trackIso_EB_;
    ecalIso2_EB_ = ecalIso_EB_;
    hcalIso2_EB_ = hcalIso_EB_;
    //
    trackIso2_EE_ = trackIso_EE_;
    ecalIso2_EE_ = ecalIso_EE_;
    hcalIso2_EE_ = hcalIso_EE_;
    //
    sihih2_EB_ = sihih_EB_;
    dphi2_EB_ = dphi_EB_;
    deta2_EB_ = deta_EB_;
    hoe2_EB_ = hoe_EB_;
    cIso2_EB_ = cIso_EB_;
    tip_bspot2_EB_ = tip_bspot_EB_;
    eop2_EB_ = eop_EB_;
    //
    sihih2_EE_ = sihih_EE_;
    dphi2_EE_ = dphi_EE_;
    deta2_EE_ = deta_EE_;
    hoe2_EE_ = hoe_EE_;
    cIso2_EE_ = cIso_EE_;
    tip_bspot2_EE_ = tip_bspot_EE_;
    eop2_EE_ = eop_EE_;
    //
    trackIsoUser2_EB_ = trackIsoUser_EB_;
    ecalIsoUser2_EB_ = ecalIsoUser_EB_;
    hcalIsoUser2_EB_ = hcalIsoUser_EB_;
    //
    trackIsoUser2_EE_ = trackIsoUser_EE_;
    ecalIsoUser2_EE_ = ecalIsoUser_EE_;
    hcalIsoUser2_EE_ = hcalIsoUser_EE_;
  }

}



ZeePlots::~ZeePlots()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ZeePlots::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
  using namespace std;
  //  cout << "In analyzer now..." << endl;
  //
  //  Get the collections here
  //
  edm::Handle<pat::CompositeCandidateCollection> ZeeCands;
  iEvent.getByLabel(zeeCollectionTag_, ZeeCands);

  if (not ZeeCands.isValid()) {
    cout << "Warning: no zee candidates in this event..." << endl;
    return;
  }
  //
  //
  const pat::CompositeCandidateCollection *zcands = ZeeCands.product();
  const pat::CompositeCandidateCollection::const_iterator 
    zeeIter = zcands->begin();
  const pat::CompositeCandidate zee = *zeeIter;
  //
  // get the parts of the composite candidate:
  const pat::Electron * myElec1=
    dynamic_cast<const pat::Electron*> (zee.daughter("electron1"));
  const pat::Electron * myElec2=
    dynamic_cast<const pat::Electron*> (zee.daughter("electron2"));
  // you can access MET like that if needed in some application
  //  const pat::MET * myMet=
  //  dynamic_cast<const pat::MET*> (zee.daughter("met"));

  //Double_t met = myMet->et();
  // extra preselection criteria
  if (usePreselection_) {
    Bool_t a1 = PassPreselectionCriteria(myElec1);
    Bool_t a2 = PassPreselectionCriteria2(myElec2);  
    if (not (a1 && a2)) return;
  }
  TLorentzVector e1;
  TLorentzVector e2;

//  math::XYZVector p1 =   myElec1->trackMomentumAtVtx();
//  math::XYZVector p2 =   myElec2->trackMomentumAtVtx();
//  e1.SetPxPyPzE(p1.X(), p1.Y(), p1.Z(), myElec1->caloEnergy());
//  e2.SetPxPyPzE(p2.X(), p2.Y(), p2.Z(), myElec2->caloEnergy());

  // Use directly the et,eta,phi from pat::Electron; assume e mass = 0.0
  e1.SetPtEtaPhiM(myElec1->et(),myElec1->eta(),myElec1->phi(),0.0);
  e2.SetPtEtaPhiM(myElec2->et(),myElec2->eta(),myElec2->phi(),0.0);

 
  TLorentzVector Z = e1+e2;
  Double_t mee = Z.M();
  // the selection plots:
  Bool_t pass = CheckCuts(myElec1) && CheckCuts2(myElec2);
  //cout << "This event passes? " << pass << ", mee is: " << mee
  //   << " and the histo is filled." << endl;
  if (not pass) return;

  h_mee->Fill(mee);
  if(fabs(e1.Eta())<1.479 && fabs(e2.Eta())<1.479)h_mee_EBEB->Fill(mee);
  if(fabs(e1.Eta())<1.479 && fabs(e2.Eta())>1.479)h_mee_EBEE->Fill(mee);
  if(fabs(e1.Eta())>1.479 && fabs(e2.Eta())<1.479)h_mee_EBEE->Fill(mee);
  if(fabs(e1.Eta())>1.479 && fabs(e2.Eta())>1.479)h_mee_EEEE->Fill(mee);
  
  h_Zcand_PT->Fill(Z.Pt());
  h_Zcand_Y->Fill(Z.Rapidity());

  h_e_PT->Fill(e1.Pt()); h_e_PT->Fill(e2.Pt()); 
  h_e_ETA->Fill(e1.Eta()); h_e_ETA->Fill(e2.Eta()); 
  h_e_PHI->Fill(e1.Phi()); h_e_PHI->Fill(e2.Phi()); 

   if(fabs(myElec1->eta())<1.479){
      h_EB_trkiso->Fill( ReturnCandVar(myElec1, 0) );
      h_EB_ecaliso->Fill(ReturnCandVar(myElec1, 1)  );
      h_EB_hcaliso->Fill(ReturnCandVar(myElec1, 2)  );
      h_EB_sIetaIeta->Fill( myElec1->scSigmaIEtaIEta() );
      h_EB_dphi->Fill( myElec1->deltaPhiSuperClusterTrackAtVtx() );
      h_EB_deta->Fill( myElec1->deltaEtaSuperClusterTrackAtVtx() );
      h_EB_HoE->Fill( myElec1->hadronicOverEm() );
    }
    else{
      h_EE_trkiso->Fill( ReturnCandVar(myElec1, 0)  );
      h_EE_ecaliso->Fill(ReturnCandVar(myElec1, 1)  );
      h_EE_hcaliso->Fill(ReturnCandVar(myElec1, 2)  );
      h_EE_sIetaIeta->Fill( myElec1->scSigmaIEtaIEta() );
      h_EE_dphi->Fill( myElec1->deltaPhiSuperClusterTrackAtVtx() );
      h_EE_deta->Fill( myElec1->deltaEtaSuperClusterTrackAtVtx() );
      h_EE_HoE->Fill( myElec1->hadronicOverEm() );
    }


 if(fabs(myElec2->eta())<1.479){
      h_EB_trkiso->Fill( ReturnCandVar(myElec2, 0) );
      h_EB_ecaliso->Fill(ReturnCandVar(myElec2, 1) );
      h_EB_hcaliso->Fill(ReturnCandVar(myElec2, 2) );
      h_EB_sIetaIeta->Fill( myElec2->scSigmaIEtaIEta() );
      h_EB_dphi->Fill( myElec2->deltaPhiSuperClusterTrackAtVtx() );
      h_EB_deta->Fill( myElec2->deltaEtaSuperClusterTrackAtVtx() );
      h_EB_HoE->Fill( myElec2->hadronicOverEm() );
    }
    else{
      h_EE_trkiso->Fill( ReturnCandVar(myElec2, 0) );
      h_EE_ecaliso->Fill(ReturnCandVar(myElec2, 1) );
      h_EE_hcaliso->Fill(ReturnCandVar(myElec2, 2) );
      h_EE_sIetaIeta->Fill( myElec2->scSigmaIEtaIEta() );
      h_EE_dphi->Fill( myElec2->deltaPhiSuperClusterTrackAtVtx() );
      h_EE_deta->Fill( myElec2->deltaEtaSuperClusterTrackAtVtx() );
      h_EE_HoE->Fill( myElec2->hadronicOverEm() );
    }




  //Double_t scEta = myElec->superCluster()->eta();
  //Double_t scPhi = myElec->superCluster()->phi();
  //Double_t scEt = myElec->superCluster()->energy()/cosh(scEta);

}


/***********************************************************************
 *
 *  Checking Cuts and making selections:
 *  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *  all the available methods take input a pointer to a  pat::Electron
 *
 *  bool  CheckCuts(const pat::Electron *): 
 *                               true if the input selection is satisfied
 *  bool  CheckCutsInverse(const pat::Electron *ele):
 *               true if the cuts with inverted the ones specified in the
 *               cfg are satisfied
 *  bool  CheckCutsNminusOne(const pat::Electron *ele, int jj):
 *               true if all the cuts with cut #jj ignored are satisfied
 *
 ***********************************************************************/
Bool_t ZeePlots::CheckCuts( const pat::Electron *ele)
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
      return fabs(val-usePrecalcIDValue_)<0.1;
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

Bool_t ZeePlots::CheckCutsInverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCutInv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCutsNminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCut(const pat::Electron *ele, int i) {
  Double_t fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return fabs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCutInv(const pat::Electron *ele, int i) {
  Double_t fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    if (InvVars_[i]) return fabs(ReturnCandVar(ele, i))>CutVars_[i];
    return fabs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  if (InvVars_[i+nBarrelVars_]) {
    if (InvVars_[i])
      return fabs(ReturnCandVar(ele, i))>CutVars_[i+nBarrelVars_];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
////////////////////////////////////////////////////////////////////////
Double_t ZeePlots::ReturnCandVar(const pat::Electron *ele, int i) {
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
// option for a second selection with the option to be used for the second
// Z leg is added - NR 09Dec09
Bool_t ZeePlots::CheckCuts2( const pat::Electron *ele)
{
  if (usePrecalcID2_) {
    if (not ele-> isElectronIDAvailable(usePrecalcIDType2_)) {
      std::cout << "Error! not existing ID with name: "
		<< usePrecalcIDType2_ << " function will return true!"
		<< std::endl;
      return true;
    }
    Double_t val = ele->electronID(usePrecalcIDType2_);
    if (usePrecalcIDSign2_ == "<") {
      return val < usePrecalcIDValue2_;
    }
    else if (usePrecalcIDSign2_ == ">") {
      return val > usePrecalcIDValue2_;
    }
    else { // equality: it returns 0,1,2,3 but as float
      return fabs(val-usePrecalcIDValue2_)<0.1;
    }
  } 
  else {
    for (int i=0; i<nBarrelVars_; ++i) {
      if (not CheckCut2(ele, i)) return false;
    }
    return true;
  }
}
//Bool_t ZeePlots::CheckCuts2( const pat::Electron *ele)
//{
//  for (int i=0; i<nBarrelVars_; ++i) {
//    if (not CheckCut2(ele, i)) return false;
//  }
//  return true;
//}
/////////////////////////////////////////////////////////////////////////

Bool_t ZeePlots::CheckCuts2Inverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCut2Inv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCuts2NminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut2(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCut2(const pat::Electron *ele, int i) {
  Double_t fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return fabs(ReturnCandVar(ele, i)) < CutVars2_[i];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars2_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
Bool_t ZeePlots::CheckCut2Inv(const pat::Electron *ele, int i) {
  Double_t fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    if (InvVars_[i]) return fabs(ReturnCandVar(ele, i))>CutVars2_[i];
    return fabs(ReturnCandVar(ele, i)) < CutVars2_[i];
  }
  if (InvVars_[i+nBarrelVars_]) {
    if (InvVars_[i])
      return fabs(ReturnCandVar(ele, i))>CutVars2_[i+nBarrelVars_];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars2_[i+nBarrelVars_];
}
//
// special preselection criteria
Bool_t ZeePlots::PassPreselectionCriteria(const pat::Electron *ele) {
  Bool_t passConvRej = true;
  Bool_t passPXB = true;
  Bool_t passEMH = true;
  if (useConversionRejection_) {
    if (ele->hasUserInt("PassConversionRejection")) {
      //std::cout << "con rej: " << ele->userInt("PassConversionRejection") << std::endl;
      if (not (ele->userInt("PassConversionRejection")==1)) passConvRej = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Conversion Rejection Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true; 
    }
  }
  if (useValidFirstPXBHit_) {
    if (ele->hasUserInt("PassValidFirstPXBHit")) {
      //std::cout << "valid1stPXB: " << ele->userInt("PassValidFirstPXBHit") << std::endl;
      if (not (ele->userInt("PassValidPXBHit")==1)) passPXB = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Valid First PXB Hit Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true;
    }
  }
  if (useExpectedMissingHits_) {
    if (ele->hasUserInt("NumberOfExpectedMissingHits")) {
      //std::cout << "missing hits: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;
      if (not (ele->userInt("NumberOfExpectedMissingHits")<=maxNumberOfExpectedMissingHits_))
        passEMH = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Number of Expected Missing Hits Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true; 
    }
  }
  return passConvRej && passPXB && passEMH;
}
//
Bool_t ZeePlots::PassPreselectionCriteria2(const pat::Electron *ele) {
  Bool_t passConvRej = true;
  Bool_t passPXB = true;
  Bool_t passEMH = true;
  if (useConversionRejection2_) {
    if (ele->hasUserInt("PassConversionRejection")) {
      //std::cout << "con rej2: " << ele->userInt("PassConversionRejection") << std::endl;
      if (not (ele->userInt("PassConversionRejection")==1)) passConvRej = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Conversion Rejection Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true; 
    }
  }
  if (useValidFirstPXBHit2_) {
    if (ele->hasUserInt("PassValidFirstPXBHit")) {
      //std::cout << "valid1stPXB2: " << ele->userInt("PassValidFirstPXBHit") << std::endl;
      if (not (ele->userInt("PassValidFirstPXBHit")==1)) passPXB = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Valid First PXB Hit Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true; 
    }
  }
  if (useExpectedMissingHits2_) {
    if (ele->hasUserInt("NumberOfExpectedMissingHits")) {
      //std::cout << "missing hits2: " << ele->userInt("NumberOfExpectedMissingHits") << std::endl;
      if (not (ele->userInt("NumberOfExpectedMissingHits")<=maxNumberOfExpectedMissingHits2_))
        passEMH = false;
    }
    else {
      std::cout << "ZeePlots: WARNING: Number of Expected Missing Hits Request Disregarded: "
                << "you must calculate it before " << std::endl;
      // return true; 
    }
  }
  return passConvRej && passPXB && passEMH;
}

////////////////////////////////////////////////////////////////////////

// ------------ method called once each job just before starting event loop  --
void 
ZeePlots::beginJob()
{
  //std::cout << "In beginJob()" << std::endl;
  h_mee      = new TH1F("h_mee",      "h_mee",       200, 0, 200);
  h_mee_EBEB = new TH1F("h_mee_EBEB", "h_mee_EBEB", 200, 0, 200);
  h_mee_EBEE = new TH1F("h_mee_EBEE", "h_mee_EBEE", 200, 0, 200);
  h_mee_EEEE = new TH1F("h_mee_EEEE", "h_mee_EEEE", 200, 0, 200);

  h_Zcand_PT = new TH1F("h_Zcand_PT", "h_Zcand_PT", 200,  0, 100);
  h_Zcand_Y  = new TH1F("h_Zcand_Y",  "h_Zcand_Y" , 200, -5, 5);

  h_e_PT  = new TH1F("h_e_PT", "h_e_PT", 200,  0, 100);
  h_e_ETA = new TH1F("h_e_ETA","h_e_ETA",200, -3, 3);
  h_e_PHI = new TH1F("h_e_PHI","h_e_PHI",200, -4, 4);


  //VALIDATION PLOTS
  //EB
  h_EB_trkiso = new TH1F("h_EB_trkiso","h_EB_trkiso",200 , 0.0, 9.0);
  h_EB_ecaliso = new TH1F("h_EB_ecaliso","h_EB_ecaliso",200, 0.0 , 9.0);
  h_EB_hcaliso = new TH1F("h_EB_hcaliso","h_EB_hcaliso",200, 0.0 , 9.0);
  h_EB_sIetaIeta = new TH1F("h_EB_sIetaIeta","h_EB_sIetaIeta",200, 0.0, 0.02 );
  h_EB_dphi = new TH1F("h_EB_dphi","h_EB_dphi",200, -0.03 , 0.03 );
  h_EB_deta = new TH1F("h_EB_deta","h_EB_deta",200, -0.01 , 0.01) ;
  h_EB_HoE = new TH1F("h_EB_HoE","h_EB_HoE",200, 0.0 , 0.2 );
  //EE
  h_EE_trkiso = new TH1F("h_EE_trkiso","h_EE_trkiso",200 , 0.0, 9.0);
  h_EE_ecaliso = new TH1F("h_EE_ecaliso","h_EE_ecaliso",200, 0.0 , 9.0);
  h_EE_hcaliso = new TH1F("h_EE_hcaliso","h_EE_hcaliso",200, 0.0 , 9.0);
  h_EE_sIetaIeta = new TH1F("h_EE_sIetaIeta","h_EE_sIetaIeta",200, 0.0, 0.1 );
  h_EE_dphi = new TH1F("h_EE_dphi","h_EE_dphi",200, -0.03 , 0.03 );
  h_EE_deta = new TH1F("h_EE_deta","h_EE_deta",200, -0.01 , 0.01) ;
  h_EE_HoE = new TH1F("h_EE_HoE","h_EE_HoE",200, 0.0 , 0.2 );





  //
  // if you add some new variable change the nBarrelVars_ accordingly
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
  // 2nd leg variables
  CutVars2_.push_back( trackIso2_EB_ );//0
  CutVars2_.push_back( ecalIso2_EB_ ); //1
  CutVars2_.push_back( hcalIso2_EB_ ); //2
  CutVars2_.push_back( sihih2_EB_ );   //3
  CutVars2_.push_back( dphi2_EB_ );    //4
  CutVars2_.push_back( deta2_EB_ );    //5
  CutVars2_.push_back( hoe2_EB_ );     //6
  CutVars2_.push_back( cIso2_EB_ );    //7
  CutVars2_.push_back( tip_bspot2_EB_);//8
  CutVars2_.push_back( eop2_EB_ );     //9
  CutVars2_.push_back( trackIsoUser2_EB_ );//10
  CutVars2_.push_back( ecalIsoUser2_EB_  );//11
  CutVars2_.push_back( hcalIsoUser2_EB_  );//12

  CutVars2_.push_back( trackIso2_EE_);//0
  CutVars2_.push_back( ecalIso2_EE_); //1
  CutVars2_.push_back( hcalIso2_EE_); //2
  CutVars2_.push_back( sihih2_EE_);   //3
  CutVars2_.push_back( dphi2_EE_);    //4
  CutVars2_.push_back( deta2_EE_);    //5
  CutVars2_.push_back( hoe2_EE_ );    //6 
  CutVars2_.push_back( cIso2_EE_ );   //7 
  CutVars2_.push_back(tip_bspot2_EE_);//8
  CutVars2_.push_back( eop2_EE_ );    //9
  CutVars2_.push_back( trackIsoUser2_EE_ );//10
  CutVars2_.push_back( ecalIsoUser2_EE_  );//11
  CutVars2_.push_back( hcalIsoUser2_EE_  );//12
  //...........................................
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
  InvVars_.push_back( cIso_EE_inv); //7
  InvVars_.push_back(tip_bspot_EE_inv);//8
  InvVars_.push_back( eop_EE_inv);     //9
  InvVars_.push_back( trackIsoUser_EE_inv );//10
  InvVars_.push_back( ecalIsoUser_EE_inv  );//11
  InvVars_.push_back( hcalIsoUser_EE_inv  );//12
  //



}

// ------------ method called once each job just after ending the event loop  -
void 
ZeePlots::endJob() {
  TFile * newfile = new TFile(TString(outputFile_),"RECREATE");
  //
  h_mee->Write();
  h_mee_EBEB->Write();
  h_mee_EBEE->Write();
  h_mee_EEEE->Write();
  h_Zcand_PT->Write();
  h_Zcand_Y->Write();

  h_e_PT->Write();
  h_e_ETA->Write();
  h_e_PHI->Write();


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
  newfile->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(ZeePlots);
