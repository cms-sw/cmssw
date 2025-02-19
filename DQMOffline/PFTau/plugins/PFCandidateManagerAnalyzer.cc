#include "DQMOffline/PFTau/plugins/PFCandidateManagerAnalyzer.h"

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



PFCandidateManagerAnalyzer::PFCandidateManagerAnalyzer(const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  PFCandidateManager( parameterSet.getParameter<double>("dRMax"),
		      parameterSet.getParameter<bool>("matchCharge"), 
		      (Benchmark::Mode) parameterSet.getParameter<int>("mode") ),
  matchLabel_( parameterSet.getParameter<InputTag>("MatchCollection") )
{
  setRange( parameterSet.getParameter<double>("ptMin"),
	    parameterSet.getParameter<double>("ptMax"),
	    parameterSet.getParameter<double>("etaMin"),
	    parameterSet.getParameter<double>("etaMax"),
	    parameterSet.getParameter<double>("phiMin"),
	    parameterSet.getParameter<double>("phiMax") );

}


void 
PFCandidateManagerAnalyzer::beginJob()
{

  BenchmarkAnalyzer::beginJob();
  setup();
}

void 
PFCandidateManagerAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  

  
  Handle<PFCandidateCollection> collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  Handle< View<Candidate> >  matchCollection;
  iEvent.getByLabel( matchLabel_, matchCollection);

  fill( *collection, *matchCollection );
}

void PFCandidateManagerAnalyzer::endJob() {
}
