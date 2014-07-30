#include "DQMOffline/PFTau/plugins/PFCandidateBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

  myColl_ = consumes< PFCandidateCollection >(inputLabel_);

}


void PFCandidateBenchmarkAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
						  edm::Run const & iRun,
						  edm::EventSetup const & iSetup )
{
  BenchmarkAnalyzer::bookHistograms(ibooker, iRun, iSetup);
  setup(ibooker);
}

void PFCandidateBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
					   const edm::EventSetup& iSetup) {
    
  Handle<PFCandidateCollection> collection; 
  iEvent.getByToken(myColl_, collection);

  fill( *collection );
}

