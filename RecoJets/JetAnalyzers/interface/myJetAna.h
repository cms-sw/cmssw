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
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// class TFile;

/****
class RBX {
  RBX();

 private:
  int nTowers;
  int ieta;
  int iphi;
  float energy;
  float time;
};

class RBXCollection {

  RBXCollection();
  void addRBX(RBX r)  {rbx_.push_back(r)};

 private:
  std::vector<RBX> rbx_;

};
*****/


class myJetAna : public edm::EDAnalyzer {

public:
  myJetAna( const edm::ParameterSet & );

private:
  void beginJob( );
  void analyze ( const edm::Event& , const edm::EventSetup& );
  void endJob();

  std::string CaloJetAlgorithm;
  std::string GenJetAlgorithm;
  edm::InputTag theTriggerResultsLabel;
  std::string JetCorrectionService;


  // --- Passed selection cuts
  TH1F *h_pt;
  TH1F *h_ptTower;
  TH1F *h_ptRBX;
  TH1F *h_ptHPD;
  TH1F *h_et;
  TH1F *h_eta;
  TH1F *h_phi;
  // ---
  
  // --- RecHits
  TH1F *HBEneOOT;
  TH1F *HEEneOOT;
  TH1F *HFEneOOT;
  TH1F *HOEneOOT;

  TH1F *HBEne;
  TH1F *HBEneTh;
  TH1F *HBEneX;
  TH1F *HBEneY;
  TH1F *HBTime;
  TH1F *HBTimeTh;
  TH1F *HBTimeX;
  TH1F *HBTimeY;
  TH1F *HEEne;
  TH1F *HEEneTh;
  TH1F *HEEneX;
  TH1F *HEEneY;
  TH1F *HEposEne;
  TH1F *HEnegEne;
  TH1F *HETime;
  TH1F *HETimeTh;
  TH1F *HETimeX;
  TH1F *HETimeY;
  TH1F *HEposTime;
  TH1F *HEnegTime;
  TH1F *HFEne;
  TH1F *HFEneTh;
  TH1F *HFTime;
  TH1F *HFTimeTh;
  TH1F *HFEneP;
  TH1F *HFTimeP;
  TH1F *HFTimePMa;
  TH1F *HFTimePM;
  TH1F *HFEneM;
  TH1F *HFTimeM;
  TH1F *HFLEne;
  TH1F *HFLTime;
  TH1F *HFSEne;
  TH2F *HFLvsS;

  TH2F *HBTvsE;
  TH2F *HETvsE;
  TH2F *HFTvsE;
  TH2F *HOTvsE;

  TH1F *HFSTime;
  TH1F *HOEne;
  TH1F *HOEneTh;
  TH1F *HOTime;
  TH1F *HOTimeTh;
  TH2F *HOocc;
  TH2F *HBocc;
  TH2F *HEocc;
  TH2F *HFocc;
  TH1F *HOSEne;
  TH1F *HOSTime;
  TH1F *HOHEne;
  TH1F *HOHTime;
  TH1F *HOHr0Ene;
  TH1F *HOHr0Time;
  TH1F *HOHrm1Ene;
  TH1F *HOHrm1Time;
  TH1F *HOHrm2Ene;
  TH1F *HOHrm2Time;
  TH1F *HOHrp1Ene;
  TH1F *HOHrp1Time;
  TH1F *HOHrp2Ene;
  TH1F *HOHrp2Time;
  TH1F *EBEne;
  TH1F *EBEneTh;
  TH1F *EBEneX;
  TH1F *EBEneY;
  TH1F *EBTime;
  TH1F *EBTimeTh;
  TH1F *EBTimeX;
  TH1F *EBTimeY;
  TH1F *EEEne;
  TH1F *EEEneTh;
  TH1F *EEEneX;
  TH1F *EEEneY;
  TH1F *EEnegEne;
  TH1F *EEposEne;
  TH1F *EETime;
  TH1F *EETimeTh;
  TH1F *EETimeX;
  TH1F *EETimeY;
  TH1F *EEnegTime;
  TH1F *EEposTime;

  TH2F *fedSize;
  TH1F *totFedSize;

  TH1F *towerHadEn;
  TH1F *towerEmEn;
  TH1F *towerOuterEn;

  TH1F *towerEmFrac;

  TH1F *RBX_et;
  TH1F *RBX_hadEnergy;
  TH1F *RBX_hcalTime;
  TH1F *RBX_nTowers;
  TH1F *RBX_N;

  TH1F *HPD_et;
  TH1F *HPD_hadEnergy;
  TH1F *HPD_hcalTime;
  TH1F *HPD_nTowers;
  TH1F *HPD_N;

  // --- from reco calomet
  TH1F *SumEt;
  TH1F *MET;
  TH1F *MET_Tower;
  TH1F *MET_RBX;
  TH1F *MET_HPD;
  TH1F *METSig;
  TH1F *MEx;
  TH1F *MEy;
  TH1F *METPhi;
  // ---

  // --- from reco vertexs
  TH1F *h_Vx;
  TH1F *h_Vy;
  TH1F *h_Vz;
  TH1F *h_VNTrks;
  // ---

  // --- from reco tracks
  TH1F *h_Trk_pt;
  TH1F *h_Trk_NTrk;
  // ---
 
  TH1F *hf_sumTowerAllEx; 
  TH1F *hf_sumTowerAllEy;
  TH1F *hf_TowerJetEt;

  TH1F *ETime; 
  TH1F *HTime; 

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

  TH1F *caloEtaEt;
  TH1F *caloEta;
  TH1F *caloPhi;

  TH1F *hitEtaEt;
  TH1F *hitEta;
  TH1F *hitPhi;

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
  TH1F *h_jet1PtHLT;

  TH1F *jetHOEne;
  TH1F *jetEMFraction;
  TH1F *NTowers;

  TH2F *h_EmEnergy;
  TH2F *h_HadEnergy;

  TH1F *st_Pt;
  TH1F *st_Constituents;
  TH1F *st_Energy;
  TH1F *st_EmEnergy;
  TH1F *st_HadEnergy;
  TH1F *st_OuterEnergy;
  TH1F *st_Eta;
  TH1F *st_Phi;
  TH1F *st_iEta;
  TH1F *st_iPhi;
  TH1F *st_Frac;

  TH2F *HFvsZ;
  TH2F *EBvHB;
  TH2F *EEvHE;
  TH2F *ECALvHCAL;
  TH2F *ECALvHCALEta1;
  TH2F *ECALvHCALEta2;
  TH2F *ECALvHCALEta3;
  TProfile *EMF_Phi;
  TProfile *EMF_Eta;
  TProfile *EMF_PhiX;
  TProfile *EMF_EtaX;
};

#endif
