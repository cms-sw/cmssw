/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/30 13:39:37 $
 *  $Revision: 1.2 $
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h" 

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

  // MET information
  theTcMETCollectionLabel       = parameters.getParameter<edm::InputTag>("TcMETCollectionLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
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
  hSumET              = dbe->book1D("METTask_SumET",     "METTask_SumET" ,500,0,2000);

}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults) {

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
    int ntrigs = triggerResults.size();
    if (_verbose) std::cout << "ntrigs=" << ntrigs << std::endl;
    
    //
    //
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    edm::TriggerNames triggerNames; // TriggerNames class
    triggerNames.init(triggerResults);
    
    //
    //
    // count number of requested Jet or MB HLT paths which have fired
    for (unsigned int i=0; i!=HLTPathsJetMBByName_.size(); i++) {
      unsigned int triggerIndex = triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
      if (triggerIndex<triggerResults.size()) {
	if (triggerResults.accept(triggerIndex)) {
	  _trig_JetMB++;
	}
      }
    }
    // for empty input vectors (n==0), take all HLT triggers!
    if (HLTPathsJetMBByName_.size()==0) _trig_JetMB=triggerResults.size()-1;

    if (_trig_JetMB==0) return;

  } else {

    edm::LogInfo("CaloMetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 
    //return;
    
  }
   
  // ==========================================================

    // **** Get the MET container  
  edm::Handle<reco::METCollection> metcoll;
  iEvent.getByLabel(theTcMETCollectionLabel, metcoll);
  
  if(!metcoll.isValid()) return;

  const METCollection *metcol = metcoll.product();
  const MET *met;
  met = &(metcol->front());
  
  LogTrace(metname)<<"[JetMETAnalyzer] Call to the TcMET analyzer";

  // ==========================================================
  // Reconstructed MET Information
  double SumET  = met->sumEt();
  double METSig = met->mEtSig();
  double Ez     = met->e_longitudinal();
  double MET    = met->pt();
  double MEx    = met->px();
  double MEy    = met->py();
  double METPhi = met->phi();

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
