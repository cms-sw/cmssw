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

//#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <vector>
#include <iostream>
#include "TFile.h"
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
  edm::InputTag wenuCollectionTag_;
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
};

#endif
