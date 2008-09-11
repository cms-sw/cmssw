#ifndef RecoExamples_myJetAna_h
#define RecoExamples_myJetAna_h
#include <TH1.h>
#include <TProfile.h>
#include <TFile.h>

/* \class myJetAna
 *
 * \author Frank Chlebana
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

// class TFile;

class myJetAna : public edm::EDAnalyzer {

public:
  myJetAna( const edm::ParameterSet & );

private:
  void beginJob( const edm::EventSetup & );
  void analyze ( const edm::Event& , const edm::EventSetup& );
  void endJob();

  std::string CaloJetAlgorithm;
  std::string GenJetAlgorithm;
  std::string JetCorrectionService;
  
  TH1F *hf_sumTowerAllEx; 
  TH1F *hf_sumTowerAllEy;
  TH1F *SumEt1;
  TH1F *MET1;
  TH1F *hf_TowerJetEt1;

  TH1F *nTowers1; 
  TH1F *nTowers2; 
  TH1F *nTowers3; 
  TH1F *nTowers4;
  TH1F *nTowersLeadJetPt1; 
  TH1F *nTowersLeadJetPt2; 
  TH1F *nTowersLeadJetPt3; 
  TH1F *nTowersLeadJetPt4;

  TH1F *totEneLeadJetEta1;
  TH1F *totEneLeadJetEta2; 
  TH1F *totEneLeadJetEta3;
  TH1F *hadEneLeadJetEta1; 
  TH1F *hadEneLeadJetEta2; 
  TH1F *hadEneLeadJetEta3;
  TH1F *emEneLeadJetEta1;  
  TH1F *emEneLeadJetEta2;  
  TH1F *emEneLeadJetEta3;

  TH1F *hadFracEta1; 
  TH1F *hadFracEta2; 
  TH1F *hadFracEta3;

  TH1F *tMassGen;

  TH1F *dijetMass1;

  TH1F *h_nCalJets1;
  TH1F *h_nGenJets1;

  TH1F *h_ptCal1;
  TH1F *h_etaCal1;
  TH1F *h_phiCal1;

  TH1F *h_ptGen1; 
  TH1F *h_etaGen1; 
  TH1F *h_phiGen1;

  TH1F *h_ptGenL1;
  TH1F *h_etaGenL1;
  TH1F *h_phiGenL1;

  TH1F *h_jetEt1;
  TH1F *h_missEt1s;
  TH1F *h_missEt1;
  TH1F *h_totMissEt1;

  TH1F *h_jet1Pt1;
  TH1F *h_jet2Pt1;

};

#endif
