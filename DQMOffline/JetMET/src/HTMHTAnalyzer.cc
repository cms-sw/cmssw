/*
 *  See header file for a description of this class.
 *
 *  \author K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/HTMHTAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h" 

#include "TVector2.h"

#include <string>
using namespace edm;

// ***********************************************************
HTMHTAnalyzer::HTMHTAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;
  _ptThreshold = 30.;

}

// ***********************************************************
HTMHTAnalyzer::~HTMHTAnalyzer() { }



void HTMHTAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & ) {
  evtCounter = 0;
  metname = "HTMHTAnalyzer";

  // PFMET information
  theJetCollectionForHTMHTLabel = parameters.getParameter<edm::InputTag>("JetCollectionForHTMHTLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  LogTrace(metname)<<"[HTMHTAnalyzer] Parameters initialization";
  ibooker.setCurrentFolder("JetMET/MET/"+_source);

  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _ptThreshold = parameters.getParameter<double>("ptThreshold");

  jetME = ibooker.book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(4,"HTMHT",1);

  hNevents = ibooker.book1D("METTask_Nevents",   "METTask_Nevents",   1,0,1);
  hNJets   = ibooker.book1D("METTask_NJets",     "METTask_NJets",     100, 0, 100);
  hMHx     = ibooker.book1D("METTask_MHx",       "METTask_MHx",       500,-500,500);
  hMHy     = ibooker.book1D("METTask_MHy",       "METTask_MHy",       500,-500,500);
  hMHT     = ibooker.book1D("METTask_MHT",       "METTask_MHT",       500,0,1000);
  hMHTPhi  = ibooker.book1D("METTask_MhTPhi",    "METTask_MhTPhi",    80,-4,4);
  hHT      = ibooker.book1D("METTask_HT",        "METTask_HT",        500,0,2000);

}

 
// ***********************************************************
void HTMHTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults) {

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
    int ntrigs = triggerResults.size();
    if (_verbose) std::cout << "ntrigs=" << ntrigs << std::endl;
    
    //
    //
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(triggerResults);
    
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

  // **** Get the Calo Jet container
  edm::Handle<reco::CaloJetCollection> jetcoll;

  // **** Get the SISCone Jet container
  iEvent.getByLabel(theJetCollectionForHTMHTLabel, jetcoll);

  if(!jetcoll.isValid()) return;

  // ==========================================================
  // Reconstructed HT & MHT Information

  int    njet=0;
  double MHx=0.;
  double MHy=0.;
  double MHT=0.;
  double MHTPhi=0.;
  double HT=0.;

  for (reco::CaloJetCollection::const_iterator calojet = jetcoll->begin(); calojet!=jetcoll->end(); ++calojet){
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
