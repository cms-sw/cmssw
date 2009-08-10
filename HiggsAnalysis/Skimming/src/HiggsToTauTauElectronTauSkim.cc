/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  This class is an EDFilter for MSSM Higgs->tautau->electronjet events
 *
 *  \author Monica Vazquez Acosta  -  Imperial College London
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToTauTauElectronTauSkim.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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


HiggsToTauTauElectronTauSkim::HiggsToTauTauElectronTauSkim(const edm::ParameterSet& iConfig) {

	// Local Debug flag
	debug            = iConfig.getParameter<bool>("DebugHiggsToTauTauElectronTauSkim");
        hltResultsLabel  = iConfig.getParameter<InputTag>("HLTResultsCollection");
        hltEventLabel    = iConfig.getParameter<InputTag>("HLTEventCollection");
        hltElectronBits  = iConfig.getParameter<std::vector<std::string> >("HLTElectronBits");
        hltFilterLabels  = iConfig.getParameter<std::vector<std::string> >("HLTFilterCollections");
	jetLabel         = iConfig.getParameter<InputTag>("JetTagCollection");
	electronLabel    = iConfig.getParameter<InputTag>("ElectronTagCollection");
	electronIdLabel  = iConfig.getParameter<InputTag>("ElectronIdTagCollection");
	minNumberOfjets  = iConfig.getParameter<int>("minNumberOfJets");
	minNumberOfelectrons  = iConfig.getParameter<int>("minNumberOfElectrons");
	jetEtMin         = iConfig.getParameter<double>("jetEtMin");
	jetEtaMin        = iConfig.getParameter<double>("jetEtaMin");
	jetEtaMax        = iConfig.getParameter<double>("jetEtaMax");
	minDRFromElectron  = iConfig.getParameter<double>("minDRFromElectron");

        nEvents         = 0;
        nSelectedEvents = 0;
}


HiggsToTauTauElectronTauSkim::~HiggsToTauTauElectronTauSkim(){
  edm::LogVerbatim("HiggsToTauTauElectronTauSkim") 
  //std::cout 
  << " Number_events_read " << nEvents
  << " Number_events_kept " << nSelectedEvents
  << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;

}


bool HiggsToTauTauElectronTauSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup ){

  using namespace trigger;

  nEvents++;


 //FIND HLT Filter objects
  edm::Handle<trigger::TriggerEvent> TriggerEventHandle;
  iEvent.getByLabel(hltEventLabel,TriggerEventHandle);

  TLorentzVector P_TriggerElectrons_temp;
  vector<TLorentzVector> P_TriggerElectrons;

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
	    P_TriggerElectrons_temp.SetPxPyPzE(TOC[KEYS[i]].px(),TOC[KEYS[i]].py(),TOC[KEYS[i]].pz(),
					     TOC[KEYS[i]].energy());
	    P_TriggerElectrons.push_back(P_TriggerElectrons_temp);	  
	  }
	}
      }
    }
  }
  
  //FIND the highest pt trigger electron
  TLorentzVector theTriggerElectron;
  double maxPt = 0;
  for(vector<TLorentzVector>::const_iterator triggerElectrons_iter = P_TriggerElectrons.begin();
      triggerElectrons_iter != P_TriggerElectrons.end(); triggerElectrons_iter++){
    if( triggerElectrons_iter->Perp() > maxPt) {
      maxPt = triggerElectrons_iter->Perp();
      theTriggerElectron.SetPxPyPzE(triggerElectrons_iter->Px(), triggerElectrons_iter->Py(), 
				  triggerElectrons_iter->Pz(), triggerElectrons_iter->E());
    }
  }
  if (maxPt == 0) return false;


  // LOOP over jets which pass cuts and are DeltaR separated to highest pt trigger electron
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
        double DR = deltaR(theTriggerElectron.Eta(),iJet->eta(),theTriggerElectron.Phi(),iJet->phi());
        if (DR > minDRFromElectron) nJets++;		
      }
    }
  }

  edm::Handle<edm::TriggerResults> HLTResults;
  iEvent.getByLabel(hltResultsLabel,HLTResults);

  //VERIFY HLT BITS
  bool ElectronTrigger = false;
  
  if (HLTResults.isValid()) {
    edm::TriggerNames triggerNames;
    triggerNames.init(*HLTResults);
    for (unsigned int iHLT = 0; iHLT < HLTResults->size(); iHLT++) {
      for (std::vector<std::string>::const_iterator iLEl = hltElectronBits.begin(); iLEl != hltElectronBits.end(); ++iLEl) {
	std::string filterElectronBit = (*iLEl);
	if(triggerNames.triggerName(iHLT) == filterElectronBit ) {
	  if ( HLTResults->accept(iHLT) ) ElectronTrigger = true;
	}
      }
    }
  }

  //ELECTRON SPECIFIC  

  int nElectrons = 0;

  if(ElectronTrigger) {
    Handle<GsfElectronCollection> electronHandle;	
    iEvent.getByLabel(electronLabel,electronHandle);


    if (electronHandle.isValid() ) {

      // Loop over electrons
      for (unsigned int i = 0; i < electronHandle->size(); i++){
	edm::Ref<reco::GsfElectronCollection> electronRef(electronHandle,i);
	//Read eID results
	std::vector<edm::Handle<edm::ValueMap<float> > > eIDValueMap(4); 
	iEvent.getByLabel( electronIdLabel , eIDValueMap[3] ); 
	const edm::ValueMap<float> & eIDmap = * eIDValueMap[3] ;
	if (eIDmap[electronRef] ) {
	  nElectrons++;
	}
      }
    }
  }
  
  
  if(ElectronTrigger) {
    if(nElectrons >=   minNumberOfelectrons && nJets >= minNumberOfjets) {
      accepted = true;
      nSelectedEvents++;
    }
  }
  else{
    if (nJets >= minNumberOfjets) {
      accepted = true;
      nSelectedEvents++;
    }	
  }
  
  return accepted;
}
