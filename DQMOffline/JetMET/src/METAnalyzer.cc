/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/12 00:21:11 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/src/METAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

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
METAnalyzer::METAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
METAnalyzer::~METAnalyzer() { }

void METAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "metAnalyzer";

  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");
  nHLTPathsJetMB_=HLTPathsJetMBByName_.size();
  HLTPathsJetMBByIndex_.resize(nHLTPathsJetMB_);

  _etThreshold = parameters.getParameter<double>("etThreshold");

  jetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(2,"MET",1);

  hNevents            = dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  hMEx                = dbe->book1D("METTask_MEx",       "METTask_MEx"   ,500,-500,500);
  hMEy                = dbe->book1D("METTask_MEy",       "METTask_MEy"   ,500,-500,500);
  hEz                 = dbe->book1D("METTask_Ez",        "METTask_Ez"    ,500,-500,500);
  hMETSig             = dbe->book1D("METTask_METSig",    "METTask_METSig",51,0,51);
  hMET                = dbe->book1D("METTask_MET",       "METTask_MET"   ,500,0,1000);
  hMETPhi             = dbe->book1D("METTask_METPhi",    "METTask_METPhi",80,-4,4);
  hSumET              = dbe->book1D("METTask_SumET",     "METTask_SumET" ,1000,0,2000);

}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults,
			      const reco::MET& met) {

  LogTrace(metname)<<"[METAnalyzer] Analyze MET";

  jetME->Fill(2);

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
  double SumET  = met.sumEt();
  double METSig = met.mEtSig();
  double Ez     = met.e_longitudinal();
  double MET    = met.pt();
  double MEx    = met.px();
  double MEy    = met.py();
  double METPhi = met.phi();

  //std::cout << _source << " " << MET << std::endl;

  //std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (MET>_etThreshold){

  hMEx->Fill(MEx);
  hMEy->Fill(MEy);
  hMET->Fill(MET);
  hMETPhi->Fill(METPhi);
  hSumET->Fill(SumET);
  hMETSig->Fill(METSig);
  hEz->Fill(Ez);

  }

}
