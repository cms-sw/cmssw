/*
 *  See header file for a description of this class.
 *
 *  $Date:$
 *  $Revision:$
 *  \author F. Chlebana - Fermilab
 */

#include "DQMOffline/JetMET/src/CaloMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <string>
using namespace std;
using namespace edm;

CaloMETAnalyzer::CaloMETAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[CaloMETAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


CaloMETAnalyzer::~CaloMETAnalyzer() { }


void CaloMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "caloMETAnalyzer";

  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/CaloMETAnalyzer");

  jetME = dbe->book1D("caloMETReco", "caloMETReco", 3, 1, 4);
  jetME->setBinLabel(1,"CaloMET",1);

  hNevents                = dbe->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
  hCaloMEx                = dbe->book1D("METTask_CaloMEx","METTask_CaloMEx",2001,-500,501);
  hCaloMEy                = dbe->book1D("METTask_CaloMEy","METTask_CaloMEy",2001,-500,501);
  hCaloEz                 = dbe->book1D("METTask_CaloEz","METTask_CaloEz",2001,-500,501);
  hCaloMETSig             = dbe->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
  hCaloMET                = dbe->book1D("METTask_CaloMET","METTask_CaloMET",2001,0,2001);
  hCaloMETPhi             = dbe->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
  hCaloSumET              = dbe->book1D("METTask_CaloSumET","METTask_CaloSumET",4001,0,4001);
  hCaloMaxEtInEmTowers    = dbe->book1D("METTask_CaloMaxEtInEmTowers","METTask_CaloMaxEtInEmTowers",4001,0,4001);
  hCaloMaxEtInHadTowers   = dbe->book1D("METTask_CaloMaxEtInHadTowers","METTask_CaloMaxEtInHadTowers",4001,0,4001);
  hCaloEtFractionHadronic = dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
  hCaloEmEtFraction       = dbe->book1D("METTask_CaloEmEtFraction","METTask_CaloEmEtFraction",100,0,1);
  hCaloHadEtInHB          = dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",4001,0,4001);
  hCaloHadEtInHO          = dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",4001,0,4001);
  hCaloHadEtInHE          = dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",4001,0,4001);
  hCaloHadEtInHF          = dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",4001,0,4001);
  hCaloHadEtInEB          = dbe->book1D("METTask_CaloHadEtInEB","METTask_CaloHadEtInEB",4001,0,4001);
  hCaloHadEtInEE          = dbe->book1D("METTask_CaloHadEtInEE","METTask_CaloHadEtInEE",4001,0,4001);
  hCaloEmEtInHF           = dbe->book1D("METTask_CaloEmEtInHF","METTask_CaloEmEtInHF",4001,0,4001);
  hCaloEmEtInEE           = dbe->book1D("METTask_CaloEmEtInEE","METTask_CaloEmEtInEE",4001,0,4001);
  hCaloEmEtInEB           = dbe->book1D("METTask_CaloEmEtInEB","METTask_CaloEmEtInEB",4001,0,4001);

}

void CaloMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::CaloMET& calomet) {

  LogTrace(metname)<<"[CaloMETAnalyzer] Analyze CaloMET";

  jetME->Fill(1);

  // ==========================================================
  // Reconstructed MET Information
  double caloSumET = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  double caloEz = calomet.e_longitudinal();
  double caloMET = calomet.pt();
  double caloMEx = calomet.px();
  double caloMEy = calomet.py();
  double caloMETPhi = calomet.phi();
  double caloMaxEtInEMTowers = calomet.maxEtInEmTowers();
  double caloMaxEtInHadTowers = calomet.maxEtInHadTowers();
  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction = calomet.emEtFraction();
  double caloHadEtInHB = calomet.hadEtInHB();
  double caloHadEtInHO = calomet.hadEtInHO();
  double caloHadEtInHE = calomet.hadEtInHE();
  double caloHadEtInHF = calomet.hadEtInHF();
  double caloEmEtInEB = calomet.emEtInEB();
  double caloEmEtInEE = calomet.emEtInEE();
  double caloEmEtInHF = calomet.emEtInHF();
  
  hCaloMEx->Fill(caloMEx);
  hCaloMEy->Fill(caloMEy);
  hCaloMET->Fill(caloMET);
  hCaloMETPhi->Fill(caloMETPhi);
  hCaloSumET->Fill(caloSumET);
  hCaloMETSig->Fill(caloMETSig);
  hCaloEz->Fill(caloEz);
  hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
  hCaloEmEtFraction->Fill(caloEmEtFraction);
  hCaloHadEtInHB->Fill(caloHadEtInHB);
  hCaloHadEtInHO->Fill(caloHadEtInHO);
  hCaloHadEtInHE->Fill(caloHadEtInHE);
  hCaloHadEtInHF->Fill(caloHadEtInHF);
  hCaloEmEtInEB->Fill(caloEmEtInEB);
  hCaloEmEtInEE->Fill(caloEmEtInEE);
  hCaloEmEtInHF->Fill(caloEmEtInHF);

}
