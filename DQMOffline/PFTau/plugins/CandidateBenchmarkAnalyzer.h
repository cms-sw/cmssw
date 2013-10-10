#ifndef __DQMOffline_PFTau_CandidateBenchmarkAnalyzer__
#define __DQMOffline_PFTau_CandidateBenchmarkAnalyzer__


#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"

#include "FWCore/Utilities/interface/EDGetToken.h"


class TH1F; 

class CandidateBenchmarkAnalyzer: public BenchmarkAnalyzer, public CandidateBenchmark {
 public:
  
  CandidateBenchmarkAnalyzer(const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob() ;
  void endJob();

  edm::EDGetTokenT< edm::View<reco::Candidate> > myColl_;
};

#endif 
