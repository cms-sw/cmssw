#ifndef HLTVertexPerformanceAnalyzer_H
#define HLTVertexPerformanceAnalyzer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <SimDataFormats/Vertex/interface/SimVertex.h>

//DQM services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/Utilities/interface/transform.h"

/** \class HLTVertexPerformanceAnalyzer
 *
 *  Code used to produce DQM validation plots for b-tag at HLT.
 *  It plots the distribution of recoVertex.z() - simVertex.z() of the primary vertex.
 */


class HLTVertexPerformanceAnalyzer : public DQMEDAnalyzer {
	public:
		explicit HLTVertexPerformanceAnalyzer(const edm::ParameterSet&);
		~HLTVertexPerformanceAnalyzer();
			void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);

	private:
		virtual void analyze(const edm::Event&, const edm::EventSetup&);
		void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

		// variables from python configuration
		edm::EDGetTokenT<edm::TriggerResults> hlTriggerResults_;
		edm::EDGetTokenT<std::vector<SimVertex> > simVertexCollection_;
		std::vector<std::string> hltPathNames_;
		HLTConfigProvider hltConfigProvider_;
		bool triggerConfChanged_;

		std::vector<edm::EDGetTokenT<reco::VertexCollection> > VertexCollection_;
		std::vector<int> hltPathIndexs_;
		
		// other class variables
		std::vector<bool> _isfoundHLTs;
		std::vector<std::string> folders;
		std::vector< std::map<std::string, MonitorElement *> > H1_;

		edm::EDConsumerBase::Labels label;
		std::vector<std::string> VertexCollection_Label;
		std::string hlTriggerResults_Label;
};

#endif
