#ifndef __DQMOffline_PFTau_PFCandidateManagerAnalyzer__
#define __DQMOffline_PFTau_PFCandidateManagerAnalyzer__


#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/PFCandidateManager.h"


class TH1F; 

class PFCandidateManagerAnalyzer: public BenchmarkAnalyzer, public PFCandidateManager {
 public:
  
  PFCandidateManagerAnalyzer(const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob() ;
  void endJob();

 private:
  edm::InputTag matchLabel_;
};

#endif 
