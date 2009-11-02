#include "DQMOffline/PFTau/plugins/CandidateBenchmarkAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using namespace reco;
using namespace edm;
using namespace std;



CandidateBenchmarkAnalyzer::CandidateBenchmarkAnalyzer(const edm::ParameterSet& parameterSet) : 
  BenchmarkAnalyzer(parameterSet),
  CandidateBenchmark( (Benchmark::Mode) parameterSet.getParameter<int>("mode") )
{}


void 
CandidateBenchmarkAnalyzer::beginJob()
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
CandidateBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup) {
  

  
  Handle< View<Candidate> > collection; 
  iEvent.getByLabel( inputLabel_, collection); 

  fill( *collection );
}

// void CandidateBenchmarkAnalyzer::endJob() {}
// {

//COLIN don't want to save several times... 
void CandidateBenchmarkAnalyzer::endJob() {
  if (outputFile_.size() != 0)
    Benchmark::DQM_->save(outputFile_);
}
