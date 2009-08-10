/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  This class is an EDFilter for MSSM Higgs->tautau->muonjet events
 *
 *  \author Monica Vazquez Acosta  -  Imperial College London
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToTauTauMuonTauSkim.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h" 


#include <TLorentzVector.h>


#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;


HiggsToTauTauMuonTauSkim::HiggsToTauTauMuonTauSkim(const edm::ParameterSet& iConfig) {

	// Local Debug flag
	debug            = iConfig.getParameter<bool>("DebugHiggsToTauTauMuonTauSkim");
        hltResultsLabel  = iConfig.getParameter<InputTag>("HLTResultsCollection");
        hltEventLabel    = iConfig.getParameter<InputTag>("HLTEventCollection");
        hltMuonBits      = iConfig.getParameter<std::vector<std::string> >("HLTMuonBits");
        hltFilterLabels  = iConfig.getParameter<std::vector<std::string> >("HLTFilterCollections");
	jetLabel         = iConfig.getParameter<InputTag>("JetTagCollection");
	minNumberOfjets  = iConfig.getParameter<int>("minNumberOfJets");
	jetEtMin         = iConfig.getParameter<double>("jetEtMin");
	jetEtaMin        = iConfig.getParameter<double>("jetEtaMin");
	jetEtaMax        = iConfig.getParameter<double>("jetEtaMax");
	minDRFromMuon  = iConfig.getParameter<double>("minDRFromMuon");

        nEvents         = 0;
        nSelectedEvents = 0;
}


HiggsToTauTauMuonTauSkim::~HiggsToTauTauMuonTauSkim(){
  edm::LogVerbatim("HiggsToTauTauMuonTauSkim") 
    //std::cout 
  << " Number_events_read " << nEvents
  << " Number_events_kept " << nSelectedEvents
  << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;

}


bool HiggsToTauTauMuonTauSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup ){

  using namespace trigger;

  nEvents++;


 //FIND HLT Filter objects
  edm::Handle<trigger::TriggerEvent> TriggerEventHandle;
  iEvent.getByLabel(hltEventLabel,TriggerEventHandle);

  TLorentzVector P_TriggerMuons_temp;
  vector<TLorentzVector> P_TriggerMuons;

  if (TriggerEventHandle.isValid()) {
    const size_type nO(TriggerEventHandle->sizeObjects());
    const TriggerObjectCollection& TOC(TriggerEventHandle->getObjects());
    for (size_type iO=0; iO!=nO; ++iO) {
    }
    const size_type nF(TriggerEventHandle->sizeFilters());
     for (size_type iF=0; iF!=nF; ++iF) {
      edm::InputTag triggerlabelname = TriggerEventHandle->filterTag(iF);
      for (std::vector<std::string>::const_iterator iL = hltFilterLabels.begin(); iL != hltFilterLabels.end(); ++iL) {
	//LOOP over right filter names
	std::string filterString = (*iL);
	if( triggerlabelname.label() == filterString ) {
	  const Keys& KEYS(TriggerEventHandle->filterKeys(iF));
	  const Vids& VIDS (TriggerEventHandle->filterIds(iF));
	  const size_type nI(VIDS.size());
	  const size_type nK(KEYS.size());
	  
	  const size_type n(std::max(nI,nK));
	  for (size_type i=0; i!=n; ++i) {
	    P_TriggerMuons_temp.SetPxPyPzE(TOC[KEYS[i]].px(),TOC[KEYS[i]].py(),TOC[KEYS[i]].pz(),
					     TOC[KEYS[i]].energy());
	    P_TriggerMuons.push_back(P_TriggerMuons_temp);	  
	  }
	}
      }
    }
  }
  
  //FIND the highest pt trigger muon
  TLorentzVector theTriggerMuon;
  double maxPt = 0;
  for(vector<TLorentzVector>::const_iterator triggerMuons_iter = P_TriggerMuons.begin();
      triggerMuons_iter != P_TriggerMuons.end(); triggerMuons_iter++){
    if( triggerMuons_iter->Perp() > maxPt) {
      maxPt = triggerMuons_iter->Perp();
      theTriggerMuon.SetPxPyPzE(triggerMuons_iter->Px(), triggerMuons_iter->Py(), 
				  triggerMuons_iter->Pz(), triggerMuons_iter->E());
    }
  }
  if (maxPt == 0) return false;


  // LOOP over jets which pass cuts and are DeltaR separated to highest pt trigger muon
  Handle<CaloJetCollection> jetHandle;	
  iEvent.getByLabel(jetLabel,jetHandle);
  int nJets = 0;
  if ( !jetHandle.isValid() ) return false;
  bool accepted = false;	
  if (jetHandle.isValid() ) {
    const reco::CaloJetCollection & jets = *(jetHandle.product());
    CaloJetCollection::const_iterator iJet;
    for (iJet = jets.begin(); iJet!= jets.end(); iJet++ ) {
      if (iJet->et()  > jetEtMin  &&
          iJet->eta() > jetEtaMin &&
	  iJet->eta() < jetEtaMax ) {
        double DR = deltaR(theTriggerMuon.Eta(),iJet->eta(),theTriggerMuon.Phi(),iJet->phi());
        if (DR > minDRFromMuon) nJets++;		
      }
    }
  }


  //VERIFY HLT BITS
  /*

  edm::Handle<edm::TriggerResults> HLTResults;
  iEvent.getByLabel(hltResultsLabel,HLTResults);

  bool MuonTrigger = false;
  
  if (HLTResults.isValid()) {
    edm::TriggerNames triggerNames;
    triggerNames.init(*HLTResults);
    for (unsigned int iHLT = 0; iHLT < HLTResults->size(); iHLT++) {
      for (std::vector<std::string>::const_iterator iLMu = hltMuonBits.begin(); iLMu != hltMuonBits.end(); ++iLMu) {
	std::string filterMuonBit = (*iLMu);
	if(triggerNames.triggerName(iHLT) == filterMuonBit ) {
	  if ( HLTResults->accept(iHLT) ) MuonTrigger = true;
	}
      }
    }
  }
  */

  if (nJets >= minNumberOfjets) {
    accepted = true;
    nSelectedEvents++;
  }	
  
  return accepted;
}
