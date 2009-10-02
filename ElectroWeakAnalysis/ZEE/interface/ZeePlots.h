#ifndef ZeePlots_H
#define ZeePlots_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
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
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      bool CheckCuts( const pat::Electron * ele);
      bool CheckCut( const pat::Electron *wenu, int i);
      bool CheckCutsInverse(const pat::Electron *ele);
      bool CheckCutInv( const pat::Electron *wenu, int i);
      bool CheckCutsNminusOne(const pat::Electron *ele, int jj);
      double ReturnCandVar(const pat::Electron *ele, int i);

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
  //
  Double_t sihih_EE_;
  Double_t deta_EE_;
  Double_t dphi_EE_;
  Double_t hoe_EE_;
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
  //
  bool sihih_EE_inv;
  bool deta_EE_inv;
  bool dphi_EE_inv;
  bool hoe_EE_inv;
  //
  int nBarrelVars_;
  //
  std::vector<Double_t> CutVars_;
  std::vector<Double_t> InvVars_;

};

#endif
