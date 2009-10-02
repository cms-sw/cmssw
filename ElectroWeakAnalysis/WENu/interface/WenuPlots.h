#ifndef WenuPlots_H
#define WenuPlots_H

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
  edm::InputTag wenuCollectionTag_;
  TFile *histofile;
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
