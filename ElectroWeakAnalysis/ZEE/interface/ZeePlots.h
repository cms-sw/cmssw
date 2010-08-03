#ifndef ZeePlots_H
#define ZeePlots_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <vector>
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "TH1F.h"
#include "TMath.h"

//
// class decleration
//

class ZeePlots : public edm::EDAnalyzer {
   public:
      explicit ZeePlots(const edm::ParameterSet&);
      ~ZeePlots();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      // for the 1st leg
      Bool_t CheckCuts( const pat::Electron * ele);
      Bool_t CheckCut( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCutsInverse(const pat::Electron *ele);
      Bool_t CheckCutInv( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCutsNminusOne(const pat::Electron *ele, Int_t jj);
      // for the 2nd leg
      Bool_t CheckCuts2( const pat::Electron * ele);
      Bool_t CheckCut2( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCuts2Inverse(const pat::Electron *ele);
      Bool_t CheckCut2Inv( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCuts2NminusOne(const pat::Electron *ele, Int_t jj);
      //
      Double_t ReturnCandVar(const pat::Electron *ele, Int_t i);
      Bool_t PassPreselectionCriteria(const pat::Electron *ele);
      Bool_t PassPreselectionCriteria2(const pat::Electron *ele);
      //
      Bool_t   useDifferentSecondLegSelection_;
      //
      // for the extra identifications and selections
      Bool_t   usePrecalcID_;
      std::string usePrecalcIDSign_;
      std::string usePrecalcIDType_;
      Double_t usePrecalcIDValue_;
      //
      Bool_t   usePrecalcID2_;
      std::string usePrecalcIDSign2_;
      std::string usePrecalcIDType2_;
      Double_t usePrecalcIDValue2_;
      //
      // for extra preselection criteria:
      Bool_t usePreselection_;
      Bool_t useValidFirstPXBHit_ ,             useValidFirstPXBHit2_,;
      Bool_t useConversionRejection_ ,          useConversionRejection2_;
      Bool_t useExpectedMissingHits_ ,          useExpectedMissingHits2_;
      Bool_t maxNumberOfExpectedMissingHits_ ,  maxNumberOfExpectedMissingHits2_;
  std::string outputFile_;
  edm::InputTag zeeCollectionTag_;
  edm::InputTag caloJetCollectionTag_;
  edm::InputTag pfJetCollectionTag_;
  TFile *histofile;
  //
  // the histograms
  TH1F *h_mee;
  TH1F *h_mee_EBEB;
  TH1F *h_mee_EBEE;
  TH1F *h_mee_EEEE;
  TH1F *h_Zcand_PT;
  TH1F *h_Zcand_Y;

  TH1F *h_e_PT;
  TH1F *h_e_ETA;
  TH1F *h_e_PHI;

  TH1F *h_EB_trkiso;
  TH1F *h_EB_ecaliso;
  TH1F *h_EB_hcaliso;
  TH1F *h_EB_sIetaIeta;
  TH1F *h_EB_dphi;
  TH1F *h_EB_deta;
  TH1F *h_EB_HoE;

  TH1F *h_EE_trkiso;
  TH1F *h_EE_ecaliso;
  TH1F *h_EE_hcaliso;
  TH1F *h_EE_sIetaIeta;
  TH1F *h_EE_dphi;
  TH1F *h_EE_deta;
  TH1F *h_EE_HoE;

  //
  TH1F *h_trackIso_eb_NmOne;
  TH1F *h_trackIso_ee_NmOne;
  // ***********************************
  //
  // the selection cuts
  Double_t trackIso_EB_ ;      
  Double_t ecalIso_EB_ ;       
  Double_t hcalIso_EB_ ;       
  // 
  Double_t trackIso_EE_ ;      
  Double_t ecalIso_EE_ ;       
  Double_t hcalIso_EE_ ;       
  //
  Double_t sihih_EB_ ;         
  Double_t deta_EB_ ;          
  Double_t dphi_EB_ ;          
  Double_t hoe_EB_ ;           
  Double_t cIso_EB_ ;          
  Double_t tip_bspot_EB_ ;     
  Double_t eop_EB_ ;           
  //
  Double_t sihih_EE_ ;         
  Double_t deta_EE_ ;          
  Double_t dphi_EE_ ;          
  Double_t hoe_EE_ ;           
  Double_t cIso_EE_ ;          
  Double_t tip_bspot_EE_ ;     
  Double_t eop_EE_ ;           
  //
  Double_t trackIsoUser_EB_ ;  
  Double_t ecalIsoUser_EB_ ;   
  Double_t hcalIsoUser_EB_ ;   
  //
  Double_t trackIsoUser_EE_ ;  
  Double_t ecalIsoUser_EE_ ;   
  Double_t hcalIsoUser_EE_ ;   
  //
  Double_t trackIso2_EB_ ;
  Double_t ecalIso2_EB_ ;
  Double_t hcalIso2_EB_ ;
  // 
  Double_t trackIso2_EE_ ;
  Double_t ecalIso2_EE_ ;
  Double_t hcalIso2_EE_ ;
  //
  Double_t sihih2_EB_ ;
  Double_t deta2_EB_ ;
  Double_t dphi2_EB_ ;
  Double_t hoe2_EB_ ;
  Double_t cIso2_EB_ ;
  Double_t tip_bspot2_EB_ ;
  Double_t eop2_EB_ ;
  //
  Double_t sihih2_EE_ ;
  Double_t deta2_EE_ ;
  Double_t dphi2_EE_ ;
  Double_t hoe2_EE_ ;
  Double_t cIso2_EE_ ;
  Double_t tip_bspot2_EE_ ;
  Double_t eop2_EE_ ;
  //
  Double_t trackIsoUser2_EB_ ;
  Double_t ecalIsoUser2_EB_ ;
  Double_t hcalIsoUser2_EB_ ;
  //
  Double_t trackIsoUser2_EE_ ;
  Double_t ecalIsoUser2_EE_ ;
  Double_t hcalIsoUser2_EE_ ;
  //
  Bool_t trackIso_EB_inv;
  Bool_t ecalIso_EB_inv;
  Bool_t hcalIso_EB_inv;
  //
  Bool_t trackIso_EE_inv;
  Bool_t ecalIso_EE_inv;
  Bool_t hcalIso_EE_inv;
  //
  Bool_t sihih_EB_inv;
  Bool_t deta_EB_inv;
  Bool_t dphi_EB_inv;
  Bool_t hoe_EB_inv;
  Bool_t cIso_EB_inv;
  Bool_t tip_bspot_EB_inv;
  Bool_t eop_EB_inv;
  //
  Bool_t sihih_EE_inv;
  Bool_t deta_EE_inv;
  Bool_t dphi_EE_inv;
  Bool_t hoe_EE_inv;
  Bool_t cIso_EE_inv;
  Bool_t tip_bspot_EE_inv;
  Bool_t eop_EE_inv;
  //
  Bool_t trackIsoUser_EB_inv;
  Bool_t ecalIsoUser_EB_inv;
  Bool_t hcalIsoUser_EB_inv;
  //
  Bool_t trackIsoUser_EE_inv;
  Bool_t ecalIsoUser_EE_inv;
  Bool_t hcalIsoUser_EE_inv;
  //
  Int_t nBarrelVars_;
  //
  std::vector<Double_t> CutVars_ , CutVars2_;
  std::vector<Bool_t> InvVars_;
  //
  // variables related to the VBTF root tuples:
  //
  Int_t runNumber, lumiSection;
  Long64_t eventNumber;
  
  Float_t ele1_sc_energy, ele1_sc_eta, ele1_sc_phi;
  Float_t ele1_sc_gsf_et;
  Float_t ele1_cand_et, ele1_cand_eta, ele1_cand_phi;
  Float_t ele1_iso_track, ele1_iso_ecal, ele1_iso_hcal;
  Float_t ele1_id_sihih, ele1_id_dphi, ele1_id_deta, ele1_id_hoe;
  Float_t ele1_cr_mhitsinner, ele1_cr_dcot, ele1_cr_dist;
  Float_t ele1_vx, ele1_vy, ele1_vz;
  
  Float_t pv_x1, pv_y1, pv_z1;

  Int_t   ele1_gsfCharge, ele1_ctfCharge, ele1_scPixCharge;
  Float_t ele1_eop, ele1_tip_bs, ele1_tip_pv;  
    
  Float_t ele2_sc_energy, ele2_sc_eta, ele2_sc_phi;
  Float_t ele2_sc_gsf_et;
  Float_t ele2_cand_et, ele2_cand_eta, ele2_cand_phi;
  Float_t ele2_iso_track, ele2_iso_ecal, ele2_iso_hcal;
  Float_t ele2_id_sihih, ele2_id_dphi, ele2_id_deta, ele2_id_hoe;
  Float_t ele2_cr_mhitsinner, ele2_cr_dcot, ele2_cr_dist;
  Float_t ele2_vx, ele2_vy, ele2_vz;

  Float_t pv_x2, pv_y2, pv_z2;

  Int_t   ele2_gsfCharge, ele2_ctfCharge, ele2_scPixCharge;
  Float_t ele2_eop, ele2_tip_bs, ele2_tip_pv;  
  
  Float_t event_caloMET, event_pfMET, event_tcMET;
  Float_t event_caloMET_phi, event_pfMET_phi, event_tcMET_phi;
  
  Float_t event_Mee;
  
  Float_t calojet_et[5];
  Float_t calojet_eta[5];
  Float_t calojet_phi[5];
  Float_t pfjet_et[5];
  Float_t pfjet_eta[5];
  Float_t pfjet_phi[5];
  Int_t event_datasetTag;

  TFile *ZEE_VBTFpreseleFile_;
  TFile *ZEE_VBTFselectionFile_;
  TTree *vbtfSele_tree;
  TTree *vbtfPresele_tree;
  std::string ZEE_VBTFselectionFileName_;
  std::string ZEE_VBTFpreseleFileName_;
  Bool_t includeJetInformationInNtuples_;
  Double_t DRJetFromElectron_;
  Int_t DatasetTag_;
};

#endif
