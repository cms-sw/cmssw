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
  userIso_EB_ = iConfig.getUntrackedParameter<Double_t>("userIso_EB", 1000.);
  //
  sihih_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih_EE");
  dphi_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi_EE");
  deta_EE_ = iConfig.getUntrackedParameter<Double_t>("deta_EE");
  hoe_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe_EE");
  userIso_EE_ = iConfig.getUntrackedParameter<Double_t>("userIso_EE", 1000.);
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
  sihih_EB_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EB_inv",
							 false);
  dphi_EB_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EB_inv",
							false);
  deta_EB_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EB_inv",
							false);
  hoe_EB_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EB_inv",
							false);
  userIso_EB_inv = iConfig.getUntrackedParameter<Bool_t>("userIso_EB_inv",
							 false);
  //
  sihih_EE_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EE_inv",
							 false);
  dphi_EE_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EE_inv",
							false);
  deta_EE_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EE_inv",
							false);
  hoe_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EE_inv",
							false);
  userIso_EE_inv = iConfig.getUntrackedParameter<Bool_t>("userIso_EE_inv",
							 false);

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
  //
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
  // some variables here
  double scEta = myElec->superCluster()->eta();
  double scPhi = myElec->superCluster()->phi();
  double scEt = myElec->superCluster()->energy()/cosh(scEta);
  double met    = myMet->et();
  double metPhi = myMet->phi();
  double mt  = sqrt(2.0*scEt*met*(1.0-(cos(scPhi)*cos(metPhi)+sin(scPhi)*sin(metPhi))));

  double trackIso = myElec->userIsolation(pat::TrackIso);
  double ecalIso = myElec->userIsolation(pat::EcalIso);
  double hcalIso = myElec->userIsolation(pat::HcalIso);
  double sihih = myElec->scSigmaIEtaIEta();
  double dphi = myElec->deltaPhiSuperClusterTrackAtVtx();
  double deta = myElec->deltaEtaSuperClusterTrackAtVtx();
  double HoE = myElec->hadronicOverEm();




  //
  //
  //
  // the inverted selection plots:
  if (CheckCutsInverse(myElec)){
    //std::cout << "-----------------INVERSION-----------passed" << std::endl;
    h_met_inverse->Fill(met);
    h_mt_inverse->Fill(mt);
    if(fabs(scEta)<1.479){
      h_met_inverse_EB->Fill(met);
      h_mt_inverse_EB->Fill(mt);
    }
    if(fabs(scEta)>1.479){
      h_met_inverse_EE->Fill(met);
      h_mt_inverse_EE->Fill(mt);
    }
  }


  ///////////////////////////////////////////////////////////////////////
  //
  // N-1 plots: plot some variable so that all the other cuts are satisfied
  if ( fabs(scEta) < 1.479) { // reminder: the precise fiducial cuts are in
                              // in the filter
    if (CheckCutsNminusOne(myElec, 0)) 
      h_trackIso_eb_NmOne->Fill(trackIso);
  }
  else {
    if (CheckCutsNminusOne(myElec, 0)) 
      h_trackIso_ee_NmOne->Fill(trackIso);
  }
  // from here on you have only events that pass the full selection
  if (not CheckCuts(myElec)) return;
  //////////////////////////////////////////////////////////////////////

  h_met->Fill(met);
  h_mt->Fill(mt);
  if(fabs(scEta)<1.479){
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
  if(fabs(scEta)>1.479){
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
  //std::cout << "tracIso: " <<  trackIso << ", " << myElec->trackIso() << ", ecaliso: " << ecalIso << ", " << myElec->ecalIso() << ", hcaliso: " << hcalIso << ", "  << myElec->hcalIso() << std::endl;

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
 *  bool  CheckCuts(const pat::Electron *): 
 *                               true if the input selection is satisfied
 *  bool  CheckCutsInverse(const pat::Electron *ele):
 *               true if the cuts with inverted the ones specified in the
 *               cfg are satisfied
 *  bool  CheckCutsNminusOne(const pat::Electron *ele, int jj):
 *               true if all the cuts with cut #jj ignored are satisfied
 *
 ***********************************************************************/
bool WenuPlots::CheckCuts( const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i) {
    if (not CheckCut(ele, i)) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////

bool WenuPlots::CheckCutsInverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCutInv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
bool WenuPlots::CheckCutsNminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
bool WenuPlots::CheckCut(const pat::Electron *ele, int i) {
  double fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return fabs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
bool WenuPlots::CheckCutInv(const pat::Electron *ele, int i) {
  double fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    if (InvVars_[i]) 
    return fabs(ReturnCandVar(ele, i))>CutVars_[i];
    return fabs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  if (InvVars_[i+nBarrelVars_]) {
    if (InvVars_[i])
      return fabs(ReturnCandVar(ele, i))>CutVars_[i+nBarrelVars_];    
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
////////////////////////////////////////////////////////////////////////
double WenuPlots::ReturnCandVar(const pat::Electron *ele, int i) {
  if (i==0) return ele->userIsolation(pat::TrackIso);
  else if (i==1) return ele->userIsolation(pat::EcalIso);
  else if (i==2) return ele->userIsolation(pat::HcalIso);
  else if (i==3) return ele->scSigmaIEtaIEta();
  else if (i==4) return ele->deltaPhiSuperClusterTrackAtVtx();
  else if (i==5) return ele->deltaEtaSuperClusterTrackAtVtx();
  else if (i==6) return ele->hadronicOverEm();
  else if (i==7) return ele->userIsolation(pat::User1Iso);
  std::cout << "Error in WenuPlots::ReturnCandVar" << std::endl;
  return -1.;

}
/////////////////////////////////////////////////////////////////////////

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
  nBarrelVars_ = 8;
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
  CutVars_.push_back( userIso_EB_ ); //7
  
  CutVars_.push_back( trackIso_EE_);//0
  CutVars_.push_back( ecalIso_EE_); //1
  CutVars_.push_back( hcalIso_EE_); //2
  CutVars_.push_back( sihih_EE_);   //3
  CutVars_.push_back( dphi_EE_);    //4
  CutVars_.push_back( deta_EE_);    //5
  CutVars_.push_back( hoe_EE_ );    //6 
  CutVars_.push_back( userIso_EE_ );//7 
  //
  InvVars_.push_back( trackIso_EB_inv);//0
  InvVars_.push_back( ecalIso_EB_inv); //1
  InvVars_.push_back( hcalIso_EB_inv); //2
  InvVars_.push_back( sihih_EB_inv);   //3
  InvVars_.push_back( dphi_EB_inv);    //4
  InvVars_.push_back( deta_EB_inv);    //5
  InvVars_.push_back( hoe_EB_inv);     //6
  InvVars_.push_back( userIso_EB_inv); //7
  //
  InvVars_.push_back( trackIso_EE_inv);//0
  InvVars_.push_back( ecalIso_EE_inv); //1
  InvVars_.push_back( hcalIso_EE_inv); //2
  InvVars_.push_back( sihih_EE_inv);   //3
  InvVars_.push_back( dphi_EE_inv);    //4
  InvVars_.push_back( deta_EE_inv);    //5
  InvVars_.push_back( hoe_EE_inv);     //6
  InvVars_.push_back( userIso_EE_inv); //7
  //


}

// ------------ method called once each job just after ending the event loop  -
void 
WenuPlots::endJob() {
  TFile * newfile = new TFile(TString(outputFile_),"RECREATE");
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
