/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  This class is an EDFilter for heavy H+->taunu events
 *
 *  \author Sami Lehti  -  HIP Helsinki
 *
 *  Updated May 25, 2009/S.Lehti
 *
 */

#include "HiggsAnalysis/Skimming/interface/HeavyChHiggsToTauNuSkim.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"

using namespace edm;
using namespace std;
using namespace reco;

HeavyChHiggsToTauNuSkim::HeavyChHiggsToTauNuSkim(const edm::ParameterSet& iConfig) {

	// Local Debug flag
	debug           = iConfig.getParameter<bool>("DebugHeavyChHiggsToTauNuSkim");

	jetLabel        = iConfig.getParameter<InputTag>("JetTagCollection");
	minNumberOfjets = iConfig.getParameter<int>("minNumberOfJets");
	jetEtMin        = iConfig.getParameter<double>("jetEtMin");
	jetEtaMin       = iConfig.getParameter<double>("jetEtaMin");
	jetEtaMax       = iConfig.getParameter<double>("jetEtaMax");

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
	  		    iJet->eta() < jetEtaMax ) nJets++;
    		}
    		if (nJets >= minNumberOfjets) {
      			accepted = true;
      			nSelectedEvents++;
    		}	
  	}

  	return accepted;
}
