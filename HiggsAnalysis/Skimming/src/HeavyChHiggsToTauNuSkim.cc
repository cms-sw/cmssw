/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  This class is an EDFilter for heavy H+->taunu events
 *
 *  \author Sami Lehti  -  HIP Helsinki
 *
 */

#include "HiggsAnalysis/Skimming/interface/HeavyChHiggsToTauNuSkim.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;


HeavyChHiggsToTauNuSkim::HeavyChHiggsToTauNuSkim(const edm::ParameterSet& iConfig) :
  HiggsAnalysisSkimType(iConfig), nEvents(0), nAccepted(0) {

	// Local Debug flag
	debug           = iConfig.getParameter<bool>("DebugHeavyChHiggsToTauNuSkim");

	jetLabel        = iConfig.getParameter<InputTag>("JetTagCollection");
	minNumberOfjets = iConfig.getUntrackedParameter<int>("minNumberOfJets",3);
	jetEtMin        = iConfig.getUntrackedParameter<double>("jetEtMin",20.);
	jetEtaMin       = iConfig.getUntrackedParameter<double>("jetEtaMin",-2.4);
	jetEtaMax       = iConfig.getUntrackedParameter<double>("jetEtaMax",2.4);
}


HeavyChHiggsToTauNuSkim::~HeavyChHiggsToTauNuSkim(){
}

void HeavyChHiggsToTauNuSkim::endJob() {
	edm::LogVerbatim("HeavyChHiggsToTauNuSkim") 
	    << "Events read " << nEvents 
            << " Events accepted " << nAccepted 
            << "\nEfficiency " << ((double)nAccepted)/((double)nEvents) 
	    << std::endl;
}

// ------------ method called to skim the data  ------------
bool HeavyChHiggsToTauNuSkim::skim(edm::Event& iEvent, const edm::EventSetup& iSetup ){

	nEvents++;

	Handle<JetTagCollection> jetTagHandle;
	try{
	    iEvent.getByLabel(jetLabel,jetTagHandle);
	}
	catch (...) {}

	bool accepted = false;
	if(jetTagHandle.isValid()){
		int nJets = 0;
		const reco::JetTagCollection & jets = *(jetTagHandle.product());
	        JetTagCollection::const_iterator iJet;
		for(iJet = jets.begin(); iJet!= jets.end(); iJet++){
	                Jet jet = *(iJet->jet().get());
			if(jet.et()  > jetEtMin  && 
			   jet.eta() > jetEtaMin &&
	                   jet.eta() < jetEtaMax ) nJets++;		
		}

		if(nJets >= minNumberOfjets) {
			accepted = true;
			nAccepted++;
		}
	}
	return accepted;
}
