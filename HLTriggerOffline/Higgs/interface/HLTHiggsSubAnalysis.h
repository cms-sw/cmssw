#ifndef HLTriggerOffline_Higgs_HLTHiggsSubAnalysis_H
#define HLTriggerOffline_Higgs_HLTHiggsSubAnalysis_H

/** \class HLTHiggsSubAnalysis
 *  Generate histograms for trigger efficiencies Higgs related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
 *
 *  $Date: 2012/03/23 11:50:56 $
 *  $Revision: 1.7 $
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
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsPlotter.h"


#include<vector>
#include<set>
#include<map>
#include<cstring>

class EVTColContainer;

class HLTHiggsSubAnalysis 
{	
       	public:
		enum
		{
			GEN,
			RECO
		};

		HLTHiggsSubAnalysis(const edm::ParameterSet & pset, 
				const std::string & analysisname );
		~HLTHiggsSubAnalysis();
	      	void beginJob();
	      	void beginRun(const edm::Run & iRun, const edm::EventSetup & iEventSetup);
	      	void analyze(const edm::Event & iEvent, const edm::EventSetup & iEventSetup, EVTColContainer * cols);

		//! Extract what objects need this analysis
		const std::vector<unsigned int> getObjectsType(const std::string & hltpath) const;

		
       	private:
		void bookobjects(const edm::ParameterSet & anpset);
		void initobjects(const edm::Event & iEvent, EVTColContainer * col);
		void InitSelector(const unsigned int & objtype);
		void insertcandidates(const unsigned int & objtype, const EVTColContainer * col,
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

		// The name of the object collections to be used in this
		// analysis. 
	      	std::string _genParticleLabel;
		std::map<unsigned int,std::string> _recLabels;
		
		//! Some kinematical parameters
	      	std::vector<double> _parametersEta;
	      	std::vector<double> _parametersPhi;
	      	std::vector<double> _parametersTurnOn;
		
		std::map<unsigned int,double> _cutMinPt;   
		std::map<unsigned int,double> _cutMaxEta;
		std::map<unsigned int,unsigned int> _cutMotherId;    //TO BE DEPRECATED (HLTMATCH)
		std::map<unsigned int,std::vector<double> > _cutsDr; // TO BE DEPRECATED (HLTMATCH)
		//! gen/rec objects cuts
		std::map<unsigned int,std::string> _genCut;
		std::map<unsigned int,std::string> _recCut;

		//! The concrete String selectors (use the string cuts introduced
		//! via the config python)
		std::map<unsigned int,StringCutObjectSelector<reco::GenParticle> *> _genSelectorMap;
	      	StringCutObjectSelector<reco::Muon>        * _recMuonSelector;
	      	StringCutObjectSelector<reco::GsfElectron> * _recElecSelector;
	      	StringCutObjectSelector<reco::CaloMET>     * _recCaloMETSelector;
	      	StringCutObjectSelector<reco::PFTau>       * _recPFTauSelector;
	      	StringCutObjectSelector<reco::Photon>      * _recPhotonSelector;
	      	StringCutObjectSelector<reco::Track>       * _recTrackSelector;
		
		// The plotters: managers of each hlt path where the plots are done
		std::vector<HLTHiggsPlotter> _analyzers;
		
		HLTConfigProvider _hltConfig;
		
	      	DQMStore* _dbe;
	      	std::map<std::string, MonitorElement *> _elements;		
};


#endif
