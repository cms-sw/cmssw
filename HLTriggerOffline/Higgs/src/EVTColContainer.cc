#ifndef HLTRIGGEROFFLINE_HIGGS_EVTCOLCONTAINER
#define HLTRIGGEROFFLINE_HIGGS_EVTCOLCONTAINER

/** \class EVTColContainer
 *  Generate histograms for trigger efficiencies Higgs related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
 *
 *  $Date: 2012/03/16 01:55:32 $
 *  $Revision: 1.1 $
 *  \author  J. Duarte Campderros
 *
 */

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h" // TO BE DEPRECATED
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsSubAnalysis.h"

#include<vector>
#include<map>

//! container with all the objects needed
struct EVTColContainer
{
	int nOfCollections;
	int nInitialized;
	const reco::GenParticleCollection * genParticles;
	const std::vector<reco::Muon> * muons;
	const std::vector<reco::GsfElectron> * electrons;
	const std::vector<reco::Photon> * photons;
	const std::vector<reco::CaloMET> * caloMETs;
	const std::vector<reco::PFTau> * pfTaus;
	const trigger::TriggerEventWithRefs * rawTriggerEvent;
	const edm::TriggerResults   * triggerResults ;
	EVTColContainer():
		nOfCollections(4),
		nInitialized(0),
		genParticles(0),
		muons(0),
		electrons(0),
		photons(0),
		pfTaus(0),
		rawTriggerEvent(0),
		triggerResults(0)
	{
	}
	//! 
	bool isAllInit()
	{
		return (nInitialized == nOfCollections);
	}

	bool isCommonInit()
	{
		return (rawTriggerEvent != 0);
	}
	//! 
	void reset()
	{
		nInitialized = 0;
		genParticles = 0;
		muons = 0; electrons = 0; photons = 0; pfTaus=0; caloMETs=0; 
		rawTriggerEvent = 0;
	}
	//! Setter: multiple overloaded function
	void set(const reco::MuonCollection * v)
	{
		muons = v;
		++nInitialized;
	}
	void set(const reco::GsfElectronCollection * v)
	{
		electrons = v;
		++nInitialized;
	}
	void set(const reco::PhotonCollection * v)
	{
		photons = v;
		++nInitialized;
	}
	void set(const reco::CaloMETCollection * v)
	{
		caloMETs = v;
		++nInitialized;
	}
	void set(const reco::PFTauCollection * v)
	{
		pfTaus = v;
		++nInitialized;
	}
	const unsigned int getSize(const unsigned int & objtype) const
	{
		unsigned int size = 0;
		if( objtype == HLTHiggsSubAnalysis::MUON && muons != 0 )
		{
			size = muons->size();
		}
		else if( objtype == HLTHiggsSubAnalysis::ELEC && electrons != 0 )
		{
			size = electrons->size();
		}
		else if( objtype == HLTHiggsSubAnalysis::PHOTON && photons != 0 )
		{
			size = photons->size();
		}
		else if( objtype == HLTHiggsSubAnalysis::CALOMET && caloMETs != 0 )
		{
			size = caloMETs->size();
		}
		else if( objtype == HLTHiggsSubAnalysis::PFTAU && pfTaus != 0 )
		{
			size = pfTaus->size();
		}

		return size;
	}
};
#endif
