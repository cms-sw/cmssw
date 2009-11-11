#ifndef __DQMOffline_PFTau_METBenchmarkAnalyzer__
#define __DQMOffline_PFTau_METBenchmarkAnalyzer__


#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/METBenchmark.h"


class TH1F; 

class METBenchmarkAnalyzer: public BenchmarkAnalyzer, public METBenchmark {
 public:
  
  METBenchmarkAnalyzer(const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob() ;
  void endJob();

};

#endif 
