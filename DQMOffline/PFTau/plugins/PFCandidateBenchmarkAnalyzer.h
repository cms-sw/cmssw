#ifndef __DQMOffline_PFTau_PFCandidateBenchmarkAnalyzer__
#define __DQMOffline_PFTau_PFCandidateBenchmarkAnalyzer__


#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/PFCandidateBenchmark.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

class TH1F; 

class PFCandidateBenchmarkAnalyzer: public BenchmarkAnalyzer, public PFCandidateBenchmark {
 public:
  
  PFCandidateBenchmarkAnalyzer(const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob();
  void endJob();

  edm::EDGetTokenT< reco::PFCandidateCollection > myColl_;
};

#endif 
