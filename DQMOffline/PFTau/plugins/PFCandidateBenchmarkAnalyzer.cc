#include "DQMOffline/PFTau/plugins/PFCandidateBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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
{}


void 
PFCandidateBenchmarkAnalyzer::beginJob()
{

  // BenchmarkAnalyzer::beginJob();
  
  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  if(!Benchmark::DQM_) {
    throw "Please initialize the DQM service in your cfg";
  }

  // part of the following could be put in the base class
  string path = "PFTask/Benchmarks/" + benchmarkLabel_ ; 
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  cout<<"path set to "<<path<<endl;
  setup();
}

void 
PFCandidateBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  

  
  Handle<PFCandidateCollection> collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  fill( *collection );
}

// void PFCandidateBenchmarkAnalyzer::endJob() {}
// {

//COLIN don't want to save several times... 
void PFCandidateBenchmarkAnalyzer::endJob() {
  if (outputFile_.size() != 0)
    Benchmark::DQM_->save(outputFile_);
}
