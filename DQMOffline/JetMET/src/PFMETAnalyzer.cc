/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/12 00:21:11 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/src/PFMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

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
PFMETAnalyzer::PFMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
PFMETAnalyzer::~PFMETAnalyzer() { }

void PFMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "pfMETAnalyzer";

  LogTrace(metname)<<"[PFMETAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");
  nHLTPathsJetMB_=HLTPathsJetMBByName_.size();
  HLTPathsJetMBByIndex_.resize(nHLTPathsJetMB_);

  _etThreshold = parameters.getParameter<double>("etThreshold");

  jetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(3,"PFMET",1);

  hNevents                = dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  hPfMEx                = dbe->book1D("METTask_PfMEx",   "METTask_PfMEx"   ,500,-500,500);
  hPfMEy                = dbe->book1D("METTask_PfMEy",   "METTask_PfMEy"   ,500,-500,500);
  hPfEz                 = dbe->book1D("METTask_PfEz",    "METTask_PfEz"    ,500,-500,500);
  hPfMETSig             = dbe->book1D("METTask_PfMETSig","METTask_PfMETSig",51,0,51);
  hPfMET                = dbe->book1D("METTask_PfMET",   "METTask_PfMET"   ,500,0,1000);
  hPfMETPhi             = dbe->book1D("METTask_PfMETPhi","METTask_PfMETPhi",80,-4,4);
  hPfSumET              = dbe->book1D("METTask_PfSumET", "METTask_PfSumET" ,1000,0,2000);

}

// ***********************************************************
void PFMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults,
			    const reco::PFMET& pfmet) {

  LogTrace(metname)<<"[PFMETAnalyzer] Analyze PFMET";

  jetME->Fill(3);

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
  double pfSumET  = pfmet.sumEt();
  double pfMETSig = pfmet.mEtSig();
  double pfEz     = pfmet.e_longitudinal();
  double pfMET    = pfmet.pt();
  double pfMEx    = pfmet.px();
  double pfMEy    = pfmet.py();
  double pfMETPhi = pfmet.phi();

  //std::cout << _source << " " << pfMET << std::endl;

  //std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (pfMET>_etThreshold){

  hPfMEx->Fill(pfMEx);
  hPfMEy->Fill(pfMEy);
  hPfMET->Fill(pfMET);
  hPfMETPhi->Fill(pfMETPhi);
  hPfSumET->Fill(pfSumET);
  hPfMETSig->Fill(pfMETSig);
  hPfEz->Fill(pfEz);

  }

}
