#include "DQMOffline/PFTau/plugins/PFCandidateBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// #include "DQMServices/Core/interface/MonitorElement.h"
// #include <TH1F.h>

using namespace reco;
using namespace edm;
using namespace std;



PFCandidateBenchmarkAnalyzer::PFCandidateBenchmarkAnalyzer(const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  PFCandidateBenchmark( (Benchmark::Mode) parameterSet.getParameter<int>("mode") )
{
  setRange( parameterSet.getParameter<double>("ptMin"),
	    parameterSet.getParameter<double>("ptMax"),
	    parameterSet.getParameter<double>("etaMin"),
	    parameterSet.getParameter<double>("etaMax"),
	    parameterSet.getParameter<double>("phiMin"),
	    parameterSet.getParameter<double>("phiMax") );

}


void 
PFCandidateBenchmarkAnalyzer::beginJob()
{
  BenchmarkAnalyzer::beginJob();
  setup();
}


void 
PFCandidateBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  

  
  Handle<PFCandidateCollection> collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  fill( *collection );
}


void PFCandidateBenchmarkAnalyzer::endJob() {
}
