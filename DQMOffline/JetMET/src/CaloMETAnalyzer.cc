/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/12 00:21:11 $
 *  $Revision: 1.9 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/src/CaloMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h" // Added temporarily by KH

#include <string>
using namespace std;
using namespace edm;

// ***********************************************************
CaloMETAnalyzer::CaloMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
CaloMETAnalyzer::~CaloMETAnalyzer() { }

void CaloMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "caloMETAnalyzer";

  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");
  nHLTPathsJetMB_=HLTPathsJetMBByName_.size();
  HLTPathsJetMBByIndex_.resize(nHLTPathsJetMB_);

  _etThreshold = parameters.getParameter<double>("etThreshold");
  _allhist     = parameters.getParameter<bool>("allHist");

  jetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(1,"CaloMET",1);

  hNevents                = dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  hCaloMEx                = dbe->book1D("METTask_CaloMEx",   "METTask_CaloMEx"   ,500,-500,500);
  hCaloMEy                = dbe->book1D("METTask_CaloMEy",   "METTask_CaloMEy"   ,500,-500,500);
  hCaloEz                 = dbe->book1D("METTask_CaloEz",    "METTask_CaloEz"    ,500,-500,500);
  hCaloMETSig             = dbe->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
  hCaloMET                = dbe->book1D("METTask_CaloMET",   "METTask_CaloMET"   ,500,0,1000);
  hCaloMETPhi             = dbe->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
  hCaloSumET              = dbe->book1D("METTask_CaloSumET", "METTask_CaloSumET" ,1000,0,2000);

  if (_allhist){
  hCaloMExLS              = dbe->book2D("METTask_CaloMEx_LS","METTask_CaloMEx_LS",200,-200,200,500,0.,500.);
  hCaloMEyLS              = dbe->book2D("METTask_CaloMEy_LS","METTask_CaloMEy_LS",200,-200,200,500,0.,500.);

  hCaloMaxEtInEmTowers    = dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,1000,0,2000);
  hCaloMaxEtInHadTowers   = dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,1000,0,2000);
  hCaloEtFractionHadronic = dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
  hCaloEmEtFraction       = dbe->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);

  hCaloHadEtInHB          = dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",1000,0,2000);
  hCaloHadEtInHO          = dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",1000,0,2000);
  hCaloHadEtInHE          = dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",1000,0,2000);
  hCaloHadEtInHF          = dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",1000,0,2000);
  hCaloEmEtInHF           = dbe->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,1000,0,2000);
  hCaloEmEtInEE           = dbe->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,1000,0,2000);
  hCaloEmEtInEB           = dbe->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,1000,0,2000);
  }

}

// ***********************************************************
void CaloMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults,
			      const reco::CaloMET& calomet) {

  LogTrace(metname)<<"[CaloMETAnalyzer] Analyze CaloMET";

  jetME->Fill(1);

  // ==========================================================  
  // Trigger information 
  //
  if(&triggerResults) {   

    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////

    //
    //
    // Check how many HLT triggers are in triggerResults 
    //int ntrigs = triggerResults.size();
    //std::cout << "ntrigs=" << ntrigs << std::endl;

    //
    //
    // Fill HLTPathsJetMBByIndex_[i]
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    edm::TriggerNames triggerNames; // TriggerNames class
    triggerNames.init(triggerResults);
    unsigned int n(nHLTPathsJetMB_);
    for (unsigned int i=0; i!=n; i++) {
      HLTPathsJetMBByIndex_[i]=triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
    }
    
    //
    //
    // for empty input vectors (n==0), use all HLT trigger paths!
    if (n==0) {
      n=triggerResults.size();
      HLTPathsJetMBByName_.resize(n);
      HLTPathsJetMBByIndex_.resize(n);
      for (unsigned int i=0; i!=n; i++) {
        HLTPathsJetMBByName_[i]=triggerNames.triggerName(i);
        HLTPathsJetMBByIndex_[i]=i;
      }
    }  

    //
    //
    // count number of requested Jet or MB HLT paths which have fired
    unsigned int fired(0);
    for (unsigned int i=0; i!=n; i++) {
      if (HLTPathsJetMBByIndex_[i]<triggerResults.size()) {
        if (triggerResults.accept(HLTPathsJetMBByIndex_[i])) {
          fired++;
        }
      }
    }

    if (fired==0) return;

  } else {

    edm::LogInfo("CaloMetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 
    //return;
    
  }
   
  // ==========================================================
  // Reconstructed MET Information
  double caloSumET  = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  double caloEz     = calomet.e_longitudinal();
  double caloMET    = calomet.pt();
  double caloMEx    = calomet.px();
  double caloMEy    = calomet.py();
  double caloMETPhi = calomet.phi();

  //std::cout << _source << " " << caloMET << std::endl;

  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction       = calomet.emEtFraction();

  double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
  double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();

  double caloHadEtInHB = calomet.hadEtInHB();
  double caloHadEtInHO = calomet.hadEtInHO();
  double caloHadEtInHE = calomet.hadEtInHE();
  double caloHadEtInHF = calomet.hadEtInHF();
  double caloEmEtInEB  = calomet.emEtInEB();
  double caloEmEtInEE  = calomet.emEtInEE();
  double caloEmEtInHF  = calomet.emEtInHF();

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  //std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (caloMET>_etThreshold){

  hCaloMEx->Fill(caloMEx);
  hCaloMEy->Fill(caloMEy);
  hCaloMET->Fill(caloMET);
  hCaloMETPhi->Fill(caloMETPhi);
  hCaloSumET->Fill(caloSumET);
  hCaloMETSig->Fill(caloMETSig);
  hCaloEz->Fill(caloEz);

  if (_allhist){
  hCaloMExLS->Fill(caloMEx,myLuminosityBlock);
  hCaloMEyLS->Fill(caloMEy,myLuminosityBlock);

  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
  hCaloEmEtFraction->Fill(caloEmEtFraction);

  hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);

  hCaloHadEtInHB->Fill(caloHadEtInHB);
  hCaloHadEtInHO->Fill(caloHadEtInHO);
  hCaloHadEtInHE->Fill(caloHadEtInHE);
  hCaloHadEtInHF->Fill(caloHadEtInHF);
  hCaloEmEtInEB->Fill(caloEmEtInEB);
  hCaloEmEtInEE->Fill(caloEmEtInEE);
  hCaloEmEtInHF->Fill(caloEmEtInHF);
  } // _allhist

  } // et threshold cut

}
