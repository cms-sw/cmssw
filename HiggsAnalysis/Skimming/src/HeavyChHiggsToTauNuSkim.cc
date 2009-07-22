/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  This class is an EDFilter for heavy H+->taunu events
 *
 *  \author Sami Lehti  -  HIP Helsinki
 *
 */

#include "HiggsAnalysis/Skimming/interface/HeavyChHiggsToTauNuSkim.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;


HeavyChHiggsToTauNuSkim::HeavyChHiggsToTauNuSkim(const edm::ParameterSet& iConfig) {

	// Local Debug flag
	debug           = iConfig.getParameter<bool>("DebugHeavyChHiggsToTauNuSkim");

	hltTauLabel	= iConfig.getParameter<InputTag>("HLTTauCollection");
	jetLabel        = iConfig.getParameter<InputTag>("JetTagCollection");
	minNumberOfjets = iConfig.getParameter<int>("minNumberOfJets");
	jetEtMin        = iConfig.getParameter<double>("jetEtMin");
	jetEtaMin       = iConfig.getParameter<double>("jetEtaMin");
	jetEtaMax       = iConfig.getParameter<double>("jetEtaMax");
	minDRFromTau    = iConfig.getParameter<double>("minDRFromTau");

        nEvents         = 0;
        nSelectedEvents = 0;

}


HeavyChHiggsToTauNuSkim::~HeavyChHiggsToTauNuSkim(){
  edm::LogVerbatim("HeavyChHiggsToTauNuSkim") 
  << " Number_events_read " << nEvents
  << " Number_events_kept " << nSelectedEvents
  << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;

}


bool HeavyChHiggsToTauNuSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup ){

  nEvents++;

  Handle<IsolatedTauTagInfoCollection> tauTagL3Handle;
  iEvent.getByLabel(hltTauLabel, tauTagL3Handle);

  if ( !tauTagL3Handle.isValid() ) return false;


  Jet theTau;
  double maxEt = 0;
  if (tauTagL3Handle.isValid() ) {
    const IsolatedTauTagInfoCollection & L3Taus = *(tauTagL3Handle.product());
    IsolatedTauTagInfoCollection::const_iterator i;
    for ( i = L3Taus.begin(); i != L3Taus.end(); i++ ) {
      if (i->discriminator() == 0) continue;
      Jet taujet = *(i->jet().get());
      if (taujet.et() > maxEt) {
        maxEt = taujet.et();
        theTau = taujet;
      }
    }
  }
	
  if (maxEt == 0) return false;

  // jets
	
  Handle<CaloJetCollection> jetHandle;	
  iEvent.getByLabel(jetLabel,jetHandle);
  
  if ( !jetHandle.isValid() ) return false;
	
  bool accepted = false;
	
  if (jetHandle.isValid() ) {
    int nJets = 0;
    const reco::CaloJetCollection & jets = *(jetHandle.product());
    CaloJetCollection::const_iterator iJet;
    for (iJet = jets.begin(); iJet!= jets.end(); iJet++ ) {
      if (iJet->et()  > jetEtMin  &&
          iJet->eta() > jetEtaMin &&
	  iJet->eta() < jetEtaMax ) {
        double DR = deltaR(theTau.eta(),iJet->eta(),theTau.phi(),iJet->phi());
        if (DR > minDRFromTau) nJets++;		
      }
    }
    if (nJets >= minNumberOfjets) {
      accepted = true;
      nSelectedEvents++;
    }	
  }
  return accepted;
}
