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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"

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
	minNumberOfjets = iConfig.getUntrackedParameter<int>("minNumberOfJets",3);
	jetEtMin        = iConfig.getUntrackedParameter<double>("jetEtMin",20.);
	jetEtaMin       = iConfig.getUntrackedParameter<double>("jetEtaMin",-2.4);
	jetEtaMax       = iConfig.getUntrackedParameter<double>("jetEtaMax",2.4);
	minDRFromTau    = iConfig.getUntrackedParameter<double>("minDRFromTau",0.5);

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

	// HLT tau

	Handle<IsolatedTauTagInfoCollection> tauTagL3Handle;
	try{
//                iEvent.getByLabel("isolatedL3SingleTau", tauTagL3Handle);
                iEvent.getByLabel(hltTauLabel, tauTagL3Handle);
        }catch(...) {;}

	Jet theTau;
	double maxEt = 0;
	if(tauTagL3Handle.isValid()){
		const IsolatedTauTagInfoCollection & L3Taus = *(tauTagL3Handle.product());
		//cout << "L3 tau collection size " << L3Taus.size() << endl;
		IsolatedTauTagInfoCollection::const_iterator i;
		for(i = L3Taus.begin(); i != L3Taus.end(); i++){
		  if(i->discriminator() == 0) continue;
		  Jet taujet = *(i->jet().get());
		  if(taujet.et() > maxEt) {
			maxEt = taujet.et();
			theTau = taujet;
		  }
		}
	}
	if(maxEt == 0) return false;

	// jets

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
	                   jet.eta() < jetEtaMax ) {
				double DR = deltaR(theTau.eta(),jet.eta(),
				                   theTau.phi(),jet.phi());
				if(DR > minDRFromTau) nJets++;		
			}
		}

		if(nJets >= minNumberOfjets) {
			accepted = true;
			nSelectedEvents++;
		}
	}
	return accepted;
}
