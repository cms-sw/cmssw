#include "DQMOffline/PFTau/plugins/METBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DQMServices/Core/interface/DQMStore.h"

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

  myColl_ = consumes< View<MET> >(inputLabel_);

}


void METBenchmarkAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & iRun,
					    edm::EventSetup const & iSetup )
{
  BenchmarkAnalyzer::bookHistograms(ibooker, iRun, iSetup);
  setup(ibooker);
}

void 
METBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup) {
  
  Handle< View<MET> > collection; 
  iEvent.getByToken(myColl_, collection);

  fill( *collection );
}

