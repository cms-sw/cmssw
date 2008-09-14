#ifndef RecoExamples_myJetAna_h
#define RecoExamples_myJetAna_h
#include <TH1.h>
#include <TH2.h>
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


  // --- Passed selection cuts
  TH1F *h_pt;
  TH1F *h_et;
  TH1F *h_eta;
  TH1F *h_phi;
  // ---
  
  TH1F *hf_sumTowerAllEx; 
  TH1F *hf_sumTowerAllEy;
  TH1F *SumEt;
  TH1F *MET;
  TH1F *hf_TowerJetEt;

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

  TH1F *dijetMass;

  TH1F *h_nCalJets;
  TH1F *h_nGenJets;

  TH1F *h_ptCal;
  TH1F *h_etaCal;
  TH1F *h_phiCal;

  TH1F *h_ptGen; 
  TH1F *h_etaGen; 
  TH1F *h_phiGen;

  TH1F *h_ptGenL;
  TH1F *h_etaGenL;
  TH1F *h_phiGenL;

  TH1F *h_jetEt;

  TH1F *h_UnclusteredEt;
  TH1F *h_UnclusteredEts;
  TH1F *h_TotalUnclusteredEt;

  TH1F *h_UnclusteredE;
  TH1F *h_TotalUnclusteredE;

  TH1F *h_ClusteredE;
  TH1F *h_TotalClusteredE;

  TH1F *h_jet1Pt;
  TH1F *h_jet2Pt;

  TH1F *EMFraction;
  TH1F *NTowers;

  TH2F *h_EmEnergy;
  TH2F *h_HadEnergy;

};

#endif
