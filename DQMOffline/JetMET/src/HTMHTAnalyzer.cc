/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/12 00:21:11 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/src/HTMHTAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h" // Added temporarily by KH

#include "TVector2.h"

#include <string>
using namespace std;
using namespace edm;

// ***********************************************************
HTMHTAnalyzer::HTMHTAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;
  _ptThreshold = 30.;

}

// ***********************************************************
HTMHTAnalyzer::~HTMHTAnalyzer() { }

void HTMHTAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "HTMHTAnalyzer";

  LogTrace(metname)<<"[HTMHTAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/MET/HTMHT");

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");
  nHLTPathsJetMB_=HLTPathsJetMBByName_.size();
  HLTPathsJetMBByIndex_.resize(nHLTPathsJetMB_);

  _ptThreshold = parameters.getParameter<double>("ptThreshold");

  jetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(4,"HTMHT",1);

  hNevents = dbe->book1D("METTask_Nevents",   "METTask_Nevents",   1,0,1);
  hNJets   = dbe->book1D("METTask_NJets",     "METTask_NJets",     100, 0, 100);
  hMHx     = dbe->book1D("METTask_MHx",       "METTask_MHx",       500,-500,500);
  hMHy     = dbe->book1D("METTask_MHy",       "METTask_MHy",       500,-500,500);
  hMHT     = dbe->book1D("METTask_MHT",       "METTask_MHT",       500,0,1000);
  hMHTPhi  = dbe->book1D("METTask_MhTPhi",    "METTask_MhTPhi",    80,-4,4);
  hHT      = dbe->book1D("METTask_HT",        "METTask_HT",        1000,0,2000);

}

// ***********************************************************
void HTMHTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults,
                              const reco::CaloJetCollection& jetcoll) {

  LogTrace(metname)<<"[HTMHTAnalyzer] Analyze HT & MHT";

  jetME->Fill(4);

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
  // Reconstructed HT & MHT Information

  int    njet=0;
  double MHx=0.;
  double MHy=0.;
  double MHT=0.;
  double MHTPhi=0.;
  double HT=0.;

  for (reco::CaloJetCollection::const_iterator calojet = jetcoll.begin(); calojet!=jetcoll.end(); ++calojet){
    if (calojet->pt()>_ptThreshold){
      njet++;
      MHx += -1.*calojet->px();
      MHy += -1.*calojet->py();
      HT  += calojet->pt();
    }
  }

  TVector2 *tv2 = new TVector2(MHx,MHy);
  MHT   =tv2->Mod();
  MHTPhi=tv2->Phi();

  //std::cout << "HTMHT " << MHT << std::endl;

  hNevents->Fill(1.);
  hNJets->Fill(njet);
  if (njet>0){
    hMHx->Fill(MHx);
    hMHy->Fill(MHy);
    hMHT->Fill(MHT);
    hMHTPhi->Fill(MHTPhi);
    hHT->Fill(HT);
  }

}
