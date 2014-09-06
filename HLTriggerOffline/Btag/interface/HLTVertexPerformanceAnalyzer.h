#ifndef HLTVertexPerformanceAnalyzer_H
#define HLTVertexPerformanceAnalyzer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/** \class HLTVertexPerformanceAnalyzer
 *
 *  Code used to produce DQM validation plots for b-tag at HLT.
 *  It plots the distribution of recoVertex.z() - simVertex.z() of the primary vertex.
 */

using namespace reco;
using namespace edm;

class HLTVertexPerformanceAnalyzer : public edm::EDAnalyzer {
	public:
		explicit HLTVertexPerformanceAnalyzer(const edm::ParameterSet&);
		~HLTVertexPerformanceAnalyzer();
		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

	private:
		virtual void beginJob() ;
		virtual void analyze(const edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;

		virtual void beginRun(edm::Run const&, edm::EventSetup const&);
		virtual void endRun(edm::Run const&, edm::EventSetup const&);
		virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

		// variables from python configuration
		InputTag hlTriggerResults_;
		std::vector<std::string> hltPathNames_;
		HLTConfigProvider hltConfigProvider_;
		bool triggerConfChanged_;

		std::vector<InputTag> VertexCollection_;
		std::vector<int> hltPathIndexs_;
		
		// other class variables
		DQMStore * dqm;
		std::vector<bool> _isfoundHLTs;
		std::vector<std::string> folders;
		std::vector< std::map<std::string, MonitorElement *> > H1_;
};

#endif
