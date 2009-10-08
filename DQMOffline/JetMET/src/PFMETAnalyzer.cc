/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/30 13:41:26 $
 *  $Revision: 1.2 $
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/PFMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

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
PFMETAnalyzer::PFMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
PFMETAnalyzer::~PFMETAnalyzer() { }

void PFMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "pfMETAnalyzer";

  // PFMET information
  thePfMETCollectionLabel       = parameters.getParameter<edm::InputTag>("PfMETCollectionLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  LogTrace(metname)<<"[PFMETAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold");

  metME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  metME->setBinLabel(3,"PFMET",1);

  hNevents              = dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  hPfMEx                = dbe->book1D("METTask_PfMEx",   "METTask_PfMEx"   ,500,-500,500);
  hPfMEy                = dbe->book1D("METTask_PfMEy",   "METTask_PfMEy"   ,500,-500,500);
  hPfEz                 = dbe->book1D("METTask_PfEz",    "METTask_PfEz"    ,500,-500,500);
  hPfMETSig             = dbe->book1D("METTask_PfMETSig","METTask_PfMETSig",51,0,51);
  hPfMET                = dbe->book1D("METTask_PfMET",   "METTask_PfMET"   ,500,0,1000);
  hPfMETPhi             = dbe->book1D("METTask_PfMETPhi","METTask_PfMETPhi",80,-4,4);
  hPfSumET              = dbe->book1D("METTask_PfSumET", "METTask_PfSumET" ,500,0,2000);

  hPfNeutralEMFraction  = dbe->book1D("METTask_PfNeutralEMFraction", "METTask_PfNeutralEMFraction" ,50,0.,1.);
  hPfNeutralHadFraction = dbe->book1D("METTask_PfNeutralHadFraction","METTask_PfNeutralHadFraction",50,0.,1.);
  hPfChargedEMFraction  = dbe->book1D("METTask_PfChargedEMFraction", "METTask_PfChargedEMFraction" ,50,0.,1.);
  hPfChargedHadFraction = dbe->book1D("METTask_PfChargedHadFraction","METTask_PfChargedHadFraction",50,0.,1.);
  hPfMuonFraction       = dbe->book1D("METTask_PfMuonFraction",      "METTask_PfMuonFraction"      ,50,0.,1.);

}

// ***********************************************************
void PFMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults) {

  LogTrace(metname)<<"[PFMETAnalyzer] Analyze PFMET";

  metME->Fill(3);

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
  edm::Handle<reco::PFMETCollection> pfmetcoll;
  iEvent.getByLabel(thePfMETCollectionLabel, pfmetcoll);
  
  if(!pfmetcoll.isValid()) return;

  const PFMETCollection *pfmetcol = pfmetcoll.product();
  const PFMET *pfmet;
  pfmet = &(pfmetcol->front());
    
  LogTrace(metname)<<"[JetMETAnalyzer] Call to the PfMET analyzer";

  // ==========================================================
  // Reconstructed MET Information
  double pfSumET  = pfmet->sumEt();
  double pfMETSig = pfmet->mEtSig();
  double pfEz     = pfmet->e_longitudinal();
  double pfMET    = pfmet->pt();
  double pfMEx    = pfmet->px();
  double pfMEy    = pfmet->py();
  double pfMETPhi = pfmet->phi();

  double pfNeutralEMFraction = pfmet->NeutralEMFraction();
  double pfNeutralHadFraction = pfmet->NeutralHadFraction();
  double pfChargedEMFraction = pfmet->ChargedEMFraction();
  double pfChargedHadFraction = pfmet->ChargedHadFraction();
  double pfMuonFraction = pfmet->MuonFraction();

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

  hPfNeutralEMFraction->Fill(pfNeutralEMFraction);
  hPfNeutralHadFraction->Fill(pfNeutralHadFraction);
  hPfChargedEMFraction->Fill(pfChargedEMFraction);
  hPfChargedHadFraction->Fill(pfChargedHadFraction);
  hPfMuonFraction->Fill(pfMuonFraction);

  }

}
