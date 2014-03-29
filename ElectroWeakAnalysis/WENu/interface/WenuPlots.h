#ifndef WenuPlots_H
#define WenuPlots_H

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
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

//#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <vector>
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TH1F.h"
#include "TMath.h"

//
// class decleration
//

class WenuPlots : public edm::EDAnalyzer {
   public:
      explicit WenuPlots(const edm::ParameterSet&);
      ~WenuPlots();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      Bool_t CheckCuts( const pat::Electron * ele);
      Bool_t CheckCut( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCutsInverse(const pat::Electron *ele);
      Bool_t CheckCutInv( const pat::Electron *wenu, Int_t i);
      Bool_t CheckCutsNminusOne(const pat::Electron *ele, Int_t jj);
      Double_t ReturnCandVar(const pat::Electron *ele, Int_t i);
      Bool_t   PassPreselectionCriteria(const pat::Electron *ele);
  // for the extra identifications and selections
  Bool_t   usePrecalcID_;
  std::string usePrecalcIDSign_;
  std::string usePrecalcIDType_;
  Double_t usePrecalcIDValue_;
  // for extra preselection criteria:
  Bool_t useValidFirstPXBHit_;
  Bool_t useConversionRejection_;
  Bool_t useExpectedMissingHits_;
  Int_t  maxNumberOfExpectedMissingHits_;
  Bool_t usePreselection_;
  std::string outputFile_;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> wenuCollectionToken_;
  edm::InputTag caloJetCollectionTag_;
  edm::EDGetTokenT< reco::CaloJetCollection > caloJetCollectionToken_;
  edm::InputTag pfJetCollectionTag_;
  edm::EDGetTokenT< reco::PFJetCollection > pfJetCollectionToken_;
  edm::EDGetTokenT< std::vector<reco::Vertex> >   PrimaryVerticesCollectionToken_;
  edm::EDGetTokenT< std::vector<reco::Vertex> >   PrimaryVerticesCollectionBSToken_;
  TFile *histofile;
  //
  //  math::XYZPoint bspotPosition_; // comment out only if you don't use pat
  //
  // the histograms ********************

  TH1F *h_met;
  TH1F *h_met_inverse;
  TH1F *h_mt;
  TH1F *h_mt_inverse;

  TH1F *h_met_EB;
  TH1F *h_met_inverse_EB;
  TH1F *h_mt_EB;
  TH1F *h_mt_inverse_EB;

  TH1F *h_met_EE;
  TH1F *h_met_inverse_EE;
  TH1F *h_mt_EE;
  TH1F *h_mt_inverse_EE;


  TH1F *h_scEt;
  TH1F *h_scEta;
  TH1F *h_scPhi;

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
  Double_t trackIso_EB_;
  Double_t ecalIso_EB_;
  Double_t hcalIso_EB_;
  //
  Double_t trackIso_EE_;
  Double_t ecalIso_EE_;
  Double_t hcalIso_EE_;
  //
  Double_t sihih_EB_;
  Double_t deta_EB_;
  Double_t dphi_EB_;
  Double_t hoe_EB_;
  Double_t cIso_EB_;
  Double_t tip_bspot_EB_;
  Double_t eop_EB_;
  //
  Double_t sihih_EE_;
  Double_t deta_EE_;
  Double_t dphi_EE_;
  Double_t hoe_EE_;
  Double_t cIso_EE_;
  Double_t tip_bspot_EE_;
  Double_t eop_EE_;
  //
  Double_t trackIsoUser_EB_;
  Double_t ecalIsoUser_EB_;
  Double_t hcalIsoUser_EB_;
  Double_t trackIsoUser_EE_;
  Double_t ecalIsoUser_EE_;
  Double_t hcalIsoUser_EE_;
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
  Bool_t trackIsoUser_EE_inv;
  Bool_t ecalIsoUser_EE_inv;
  Bool_t hcalIsoUser_EE_inv;
  //
  //
  Int_t nBarrelVars_;
  //
  std::vector<Double_t> CutVars_;
  std::vector<Bool_t> InvVars_;
  //
  // variables related to the VBTF root tuples:
  //
  Int_t runNumber, lumiSection;
  Long64_t eventNumber;
  Float_t ele_sc_energy, ele_sc_eta, ele_sc_phi, ele_sc_rho;
  Float_t ele_sc_gsf_et;
  Float_t ele_cand_et, ele_cand_eta, ele_cand_phi;
  Float_t ele_iso_track, ele_iso_ecal, ele_iso_hcal;
  Float_t ele_id_sihih, ele_id_dphi, ele_id_deta, ele_id_hoe;
  Float_t ele_cr_dcot, ele_cr_dist;
  Int_t   ele_cr_mhitsinner;
  Float_t ele_vx, ele_vy, ele_vz;
  Float_t ele_pin, ele_pout;
  Float_t pv_x, pv_y, pv_z;
  Int_t   ele_gsfCharge, ele_ctfCharge, ele_scPixCharge;
  Float_t ele_eop, ele_tip_bs, ele_tip_pv;
  Float_t event_caloMET, event_pfMET, event_tcMET;
  Float_t event_caloSumEt, event_pfSumEt, event_tcSumEt;
  Float_t event_caloMET_phi, event_pfMET_phi, event_tcMET_phi;
  Float_t event_caloMT, event_pfMT, event_tcMT;
  Float_t calojet_et[5];
  Float_t calojet_eta[5];
  Float_t calojet_phi[5];
  Float_t pfjet_et[5];
  Float_t pfjet_eta[5];
  Float_t pfjet_phi[5];
  Float_t ele2nd_sc_gsf_et;
  Float_t ele2nd_sc_eta;
  Float_t ele2nd_sc_phi;
  Float_t ele2nd_sc_rho;
  Float_t ele2nd_cand_eta;
  Float_t ele2nd_cand_phi;
  Float_t ele2nd_pin;
  Float_t ele2nd_pout;
  Int_t   ele2nd_passes_selection;
  Int_t   ele2nd_ecalDriven;
  Float_t ele_hltmatched_dr;
  Int_t   event_triggerDecision;
  Int_t event_datasetTag;

  TFile *WENU_VBTFpreseleFile_;
  TFile *WENU_VBTFselectionFile_;
  TTree *vbtfSele_tree;
  TTree *vbtfPresele_tree;
  std::string WENU_VBTFselectionFileName_;
  std::string WENU_VBTFpreseleFileName_;
  Bool_t includeJetInformationInNtuples_;
  Bool_t storeExtraInformation_;
  Double_t DRJetFromElectron_;
  Int_t DatasetTag_;
  // for the 2nd electron storage
  Bool_t storeAllSecondElectronVariables_;
  Float_t ele2nd_cand_et;
  Float_t ele2nd_iso_track, ele2nd_iso_ecal, ele2nd_iso_hcal;
  Float_t ele2nd_id_sihih, ele2nd_id_deta, ele2nd_id_dphi, ele2nd_id_hoe;
  Float_t ele2nd_cr_dcot, ele2nd_cr_dist;
  Float_t ele2nd_vx, ele2nd_vy, ele2nd_vz;
  Int_t   ele2nd_cr_mhitsinner, ele2nd_gsfCharge, ele2nd_ctfCharge, ele2nd_scPixCharge;
  Float_t ele2nd_eop, ele2nd_tip_bs, ele2nd_tip_pv;
  Float_t ele2nd_hltmatched_dr;
  std::vector<Int_t> VtxTracksSize;
  std::vector<Float_t> VtxNormalizedChi2;
  std::vector<Int_t> VtxTracksSizeBS;
  std::vector<Float_t> VtxNormalizedChi2BS;
  Float_t pvbs_x, pvbs_y, pvbs_z;
  Float_t ele_tip_pvbs, ele2nd_tip_pvbs;
};

#endif
