#include "DQMOffline/PFTau/plugins/METBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/MET.h"


using namespace reco;
using namespace edm;
using namespace std;



METBenchmarkAnalyzer::METBenchmarkAnalyzer(const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  METBenchmark( (Benchmark::Mode) parameterSet.getParameter<int>("mode") )
{

  setRange( parameterSet.getParameter<double>("ptMin"),
	    parameterSet.getParameter<double>("ptMax"),
	    -0.1, 0.1, // range in eta for MET. 
	    parameterSet.getParameter<double>("phiMin"),
	    parameterSet.getParameter<double>("phiMax") );
}


void 
METBenchmarkAnalyzer::beginJob()
{

  BenchmarkAnalyzer::beginJob();
  setup();
}

void 
METBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  

  
  Handle< View<MET> > collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  fill( *collection );
}


void METBenchmarkAnalyzer::endJob() {
}
