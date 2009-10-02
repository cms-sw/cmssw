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
  //
  sihih_EE_ = iConfig.getUntrackedParameter<Double_t>("sihih_EE");
  dphi_EE_ = iConfig.getUntrackedParameter<Double_t>("dphi_EE");
  deta_EE_ = iConfig.getUntrackedParameter<Double_t>("deta_EE");
  hoe_EE_ = iConfig.getUntrackedParameter<Double_t>("hoe_EE");
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
  //
  sihih_EE_inv = iConfig.getUntrackedParameter<Bool_t>("sihih_EE_inv", false);
  dphi_EE_inv = iConfig.getUntrackedParameter<Bool_t>("dphi_EE_inv", false);
  deta_EE_inv = iConfig.getUntrackedParameter<Bool_t>("deta_EE_inv",false);
  hoe_EE_inv = iConfig.getUntrackedParameter<Bool_t>("hoe_EE_inv",false);


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

  //double met = myMet->et();
  math::XYZVector p1 =   myElec1->trackMomentumAtVtx();
  math::XYZVector p2 =   myElec2->trackMomentumAtVtx();
  TLorentzVector e1;
  TLorentzVector e2;
  e1.SetPxPyPzE(p1.X(), p1.Y(), p1.Z(), myElec1->caloEnergy());
  e2.SetPxPyPzE(p2.X(), p2.Y(), p2.Z(), myElec2->caloEnergy());
  TLorentzVector Z = e1+e2;
  double mee = Z.M();
  // the selection plots:
  bool pass = CheckCuts(myElec1) && CheckCuts(myElec2);
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




  //double scEta = myElec->superCluster()->eta();
  //double scPhi = myElec->superCluster()->phi();
  //double scEt = myElec->superCluster()->energy()/cosh(scEta);

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
bool ZeePlots::CheckCuts( const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i) {
    if (not CheckCut(ele, i)) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////

bool ZeePlots::CheckCutsInverse(const pat::Electron *ele)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if ( CheckCutInv(ele, i) == false) return false;
  }
  return true;

}
/////////////////////////////////////////////////////////////////////////
bool ZeePlots::CheckCutsNminusOne(const pat::Electron *ele, int jj)
{
  for (int i=0; i<nBarrelVars_; ++i){
    if (i==jj) continue;
    if ( CheckCut(ele, i) == false) return false;
  }
  return true;
}
/////////////////////////////////////////////////////////////////////////
bool ZeePlots::CheckCut(const pat::Electron *ele, int i) {
  double fabseta = fabs(ele->superCluster()->eta());
  if ( fabseta<1.479) {
    return fabs(ReturnCandVar(ele, i)) < CutVars_[i];
  }
  return fabs(ReturnCandVar(ele, i)) < CutVars_[i+nBarrelVars_];
}
/////////////////////////////////////////////////////////////////////////
bool ZeePlots::CheckCutInv(const pat::Electron *ele, int i) {
  double fabseta = fabs(ele->superCluster()->eta());
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
double ZeePlots::ReturnCandVar(const pat::Electron *ele, int i) {
  if (i==0) return ele->trackIso();
  else if (i==1) return ele->ecalIso();
  else if (i==2) return ele->hcalIso();
  else if (i==3) return ele->scSigmaIEtaIEta();
  else if (i==4) return ele->deltaEtaSuperClusterTrackAtVtx();
  else if (i==5) return ele->deltaPhiSuperClusterTrackAtVtx();
  else if (i==6) return ele->hadronicOverEm();
  std::cout << "Error in ZeePlots::ReturnCandVar" << std::endl;
  return -1.;

}
/////////////////////////////////////////////////////////////////////////



// ------------ method called once each job just before starting event loop  --
void 
ZeePlots::beginJob(const edm::EventSetup&)
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

  //
  // if you add some new variable change the nBarrelVars_ accordingly
  nBarrelVars_ = 7;
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

  CutVars_.push_back( trackIso_EE_);//0
  CutVars_.push_back( ecalIso_EE_); //1
  CutVars_.push_back( hcalIso_EE_); //2
  CutVars_.push_back( sihih_EE_);   //3
  CutVars_.push_back( dphi_EE_);    //4
  CutVars_.push_back( deta_EE_);    //5
  CutVars_.push_back( hoe_EE_ );    //6 
  //
  InvVars_.push_back( trackIso_EB_inv);//0
  InvVars_.push_back( ecalIso_EB_inv); //1
  InvVars_.push_back( hcalIso_EB_inv); //2
  InvVars_.push_back( sihih_EB_inv);   //3
  InvVars_.push_back( dphi_EB_inv);    //4
  InvVars_.push_back( deta_EB_inv);    //5
  InvVars_.push_back( hoe_EB_inv);     //6
  //
  InvVars_.push_back( trackIso_EE_inv);//0
  InvVars_.push_back( ecalIso_EE_inv); //1
  InvVars_.push_back( hcalIso_EE_inv); //2
  InvVars_.push_back( sihih_EE_inv);   //3
  InvVars_.push_back( dphi_EE_inv);    //4
  InvVars_.push_back( deta_EE_inv);    //5
  InvVars_.push_back( hoe_EE_inv);     //6

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
  //
  newfile->Close();

}


//define this as a plug-in
DEFINE_FWK_MODULE(ZeePlots);
