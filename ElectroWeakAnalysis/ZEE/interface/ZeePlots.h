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
      bool CheckCuts( const pat::Electron * ele);
      bool CheckCut( const pat::Electron *wenu, int i);
      bool CheckCutsInverse(const pat::Electron *ele);
      bool CheckCutInv( const pat::Electron *wenu, int i);
      bool CheckCutsNminusOne(const pat::Electron *ele, int jj);
      // for the 2nd leg
      bool CheckCuts2( const pat::Electron * ele);
      bool CheckCut2( const pat::Electron *wenu, int i);
      bool CheckCuts2Inverse(const pat::Electron *ele);
      bool CheckCut2Inv( const pat::Electron *wenu, int i);
      bool CheckCuts2NminusOne(const pat::Electron *ele, int jj);
      //
      double ReturnCandVar(const pat::Electron *ele, int i);
      bool   useDifferentSecondLegSelection_;
  std::string outputFile_;
  edm::InputTag zeeCollectionTag_;
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
  Double_t userIso_EB_;
  //
  Double_t sihih_EE_;
  Double_t deta_EE_;
  Double_t dphi_EE_;
  Double_t hoe_EE_;
  Double_t userIso_EE_;
  //
  Double_t trackIso2_EB_;
  Double_t ecalIso2_EB_;
  Double_t hcalIso2_EB_;
  //
  Double_t trackIso2_EE_;
  Double_t ecalIso2_EE_;
  Double_t hcalIso2_EE_;
  //
  Double_t sihih2_EB_;
  Double_t deta2_EB_;
  Double_t dphi2_EB_;
  Double_t hoe2_EB_;
  Double_t userIso2_EB_;
  //
  Double_t sihih2_EE_;
  Double_t deta2_EE_;
  Double_t dphi2_EE_;
  Double_t hoe2_EE_;
  Double_t userIso2_EE_;
  //
  bool trackIso_EB_inv;
  bool ecalIso_EB_inv;
  bool hcalIso_EB_inv;
  //
  bool trackIso_EE_inv;
  bool ecalIso_EE_inv;
  bool hcalIso_EE_inv;
  //
  bool sihih_EB_inv;
  bool deta_EB_inv;
  bool dphi_EB_inv;
  bool hoe_EB_inv;
  bool userIso_EB_inv;
  //
  bool sihih_EE_inv;
  bool deta_EE_inv;
  bool dphi_EE_inv;
  bool hoe_EE_inv;
  bool userIso_EE_inv;
  //
  int nBarrelVars_;
  //
  std::vector<Double_t> CutVars_;
  std::vector<Double_t> CutVars2_;
  std::vector<Double_t> InvVars_;

};

#endif
