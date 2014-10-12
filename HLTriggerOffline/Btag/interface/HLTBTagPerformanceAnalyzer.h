#ifndef HLTBTagPerformanceAnalyzer_H
#define HLTBTagPerformanceAnalyzer_H

// system include files
#include <memory>
#include <string>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//DQM services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

//for gen matching 
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include <Math/GenVector/VectorUtil.h>

#include "FWCore/Utilities/interface/transform.h"

/** \class HLTBTagPerformanceAnalyzer
 *
 *  Code used to produce DQM validation plots for b-tag at HLT.
 *  It plots the 1D distribution of the b-tag discriminant for all,b,c,lights,g jets
 *  And it plots the 2D distribution of the b-tag discriminant for all,b,c,lights,g jets vs pT 
 */
 
using namespace reco;
using namespace edm;

class HLTBTagPerformanceAnalyzer : public DQMEDAnalyzer { 
		public:
			explicit HLTBTagPerformanceAnalyzer(const edm::ParameterSet&);
			~HLTBTagPerformanceAnalyzer();
			void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);

		private:
			virtual void analyze(const edm::Event&, const edm::EventSetup&);
			void bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &  iSetup ) override;

		struct JetRefCompare :
			public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
				inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
						const edm::RefToBase<reco::Jet> &j2) const
				{ return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
			};

		typedef std::map<edm::RefToBase<reco::Jet>, float, JetRefCompare> JetTagMap;

		// variables from python configuration
		edm::EDGetTokenT<TriggerResults> hlTriggerResults_;
		std::vector<std::string> hltPathNames_;
		HLTConfigProvider hltConfigProvider_;
		bool triggerConfChanged_;
		std::vector<edm::EDGetTokenT<reco::JetTagCollection> > JetTagCollection_;

		/// other class variable
		std::vector<bool> _isfoundHLTs;
		std::vector<int> hltPathIndexs_;

		// gen level tag-handlers
		typedef unsigned int            flavour_t;
		typedef std::vector<flavour_t>  flavours_t;

		edm::EDGetTokenT<JetFlavourMatchingCollection>						m_mcPartons;        // MC truth match - jet association to partons
		std::vector<std::string>  											m_mcLabels;         // MC truth match - labels
		std::vector<flavours_t>   											m_mcFlavours;       // MC truth match - flavours selection
		double                    											m_mcRadius;         // MC truth match - deltaR association radius
		bool                      											m_mcMatching;       // MC truth matching anabled/disabled

		/// DQM folder handle
		std::vector<std::string> folders;

		// Histogram handler
		std::vector< std::map<std::string, MonitorElement *> > H1_;
		std::vector< std::map<std::string, MonitorElement *> > H2_;

		// Other variables
		edm::EDConsumerBase::Labels label;
		std::string m_mcPartons_Label;
		std::vector<std::string> JetTagCollection_Label;
		std::string hlTriggerResults_Label;
		std::string hltConfigProvider_Label;

};


#endif


