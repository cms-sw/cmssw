#include "DQMOffline/PFTau/plugins/MatchMETBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/MET.h"

using namespace reco;
using namespace edm;
using namespace std;

MatchMETBenchmarkAnalyzer::MatchMETBenchmarkAnalyzer(const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  MatchMETBenchmark( (Benchmark::Mode) parameterSet.getParameter<int>("mode") )
{
  matchedinputLabel_=parameterSet.getParameter<edm::InputTag>("MatchCollection");
//  setRange( parameterSet.getParameter<double>("ptMin"),
//	    parameterSet.getParameter<double>("ptMax"),
//	    -0.1, 0.1, // range in eta for MET. 
//	    parameterSet.getParameter<double>("phiMin"),
//	    parameterSet.getParameter<double>("phiMax") );
}

void 
MatchMETBenchmarkAnalyzer::beginJob()
{

  BenchmarkAnalyzer::beginJob();
  setup();
}

void 
MatchMETBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  
  
  Handle< View<MET> > collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  Handle< View<MET> > matchedcollection; 
  iEvent.getByLabel( matchedinputLabel_, matchedcollection); 

  fillOne( (*collection)[0] , (*matchedcollection)[0]);
}

void MatchMETBenchmarkAnalyzer::endJob() {
}
