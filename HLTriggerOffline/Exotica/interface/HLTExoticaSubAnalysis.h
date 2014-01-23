#ifndef HLTriggerOffline_Exotica_HLTExoticaSubAnalysis_H
#define HLTriggerOffline_Exotica_HLTExoticaSubAnalysis_H

/** \class HLTExoticaSubAnalysis
 *  Generate histograms for trigger efficiencies Exotica related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/ExoticaWGHLTValidate
 *
 *  \author  J. Duarte Campderros (based and adapted on J. Klukas,
 *           M. Vander Donckt and J. Alcaraz code from the 
 *           HLTriggerOffline/Muon package)
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaPlotter.h"


#include<vector>
#include<set>
#include<map>
#include<cstring>

class EVTColContainer;

class HLTExoticaSubAnalysis 
{	
       	public:
		enum
		{
			GEN,
			RECO
		};

		HLTExoticaSubAnalysis(const edm::ParameterSet & pset, 
				const std::string & analysisname );
		~HLTExoticaSubAnalysis();
	      	void beginJob();
	      	void beginRun(const edm::Run & iRun, const edm::EventSetup & iEventSetup);
	      	void analyze(const edm::Event & iEvent, const edm::EventSetup & iEventSetup, EVTColContainer * cols);

		//! Return the objects (muons,electrons,photons,...) needed by a HLT path.
		//! Will in general return: 0 for muon, 1 for electron, 2 for photon,
		//! 3 for PFMET, for for PFTau, 6 for Jet.
		//! Notice that this function is really based on a parsing of the name of
		//! the path; any incongruences theremay lead to problems.
		const std::vector<unsigned int> getObjectsType(const std::string & hltpath) const;

		
       	private:
       	//! Books the maps, telling which collection should come from witch label
       	void bookobjects(const edm::ParameterSet & anpset);
       	//! Gets the collections booked
		void initobjects(const edm::Event & iEvent, EVTColContainer * col);
		void initSelector(const unsigned int & objtype);
		void insertCandidates(const unsigned int & objtype, const EVTColContainer * col,
				std::vector<MatchStruct> * matches);

		void bookHist(const std::string & source, const std::string & objType,
			       	const std::string & variable);
		void fillHist(const std::string & source,const std::string & objType, 
				const std::string & variable, const float & value );

		edm::ParameterSet _pset;

		std::string _analysisname;

		//! The minimum number of reco/gen candidates needed by the analysis
		unsigned int _minCandidates;

		std::string _hltProcessName;
		
		//! the hlt paths with regular expressions
		std::vector<std::string> _hltPathsToCheck;
		//! the hlt paths found in the hltConfig
		std::set<std::string> _hltPaths;

		//! Relation between the short version of a path 
		std::map<std::string,std::string> _shortpath2long;

		// The name of the object collections to be used in this analysis. 
		std::string _genParticleLabel;
		std::map<unsigned int,std::string> _recLabels;
		
		//! Some kinematical parameters
		std::vector<double> _parametersEta;
		std::vector<double> _parametersPhi;
		std::vector<double> _parametersTurnOn;
		
		//! gen/rec objects cuts
		std::map<unsigned int,std::string> _genCut;
		std::map<unsigned int,std::string> _recCut;

		//! The concrete String selectors (use the string cuts introduced
		//! via the config python)
		std::map<unsigned int,StringCutObjectSelector<reco::GenParticle> *> _genSelectorMap;
	   	StringCutObjectSelector<reco::Muon>        * _recMuonSelector;
	   	StringCutObjectSelector<reco::GsfElectron> * _recElecSelector;
	   	StringCutObjectSelector<reco::PFMET>       * _recPFMETSelector;
	   	StringCutObjectSelector<reco::PFTau>       * _recPFTauSelector;
	   	StringCutObjectSelector<reco::Photon>      * _recPhotonSelector;
	   	StringCutObjectSelector<reco::PFJet>       * _recJetSelector;
		
		// The plotters: managers of each hlt path where the plots are done
		std::vector<HLTExoticaPlotter> _analyzers;
		
		HLTConfigProvider _hltConfig;
		
	   	DQMStore* _dbe;
	   	std::map<std::string, MonitorElement *> _elements;		
};


#endif
