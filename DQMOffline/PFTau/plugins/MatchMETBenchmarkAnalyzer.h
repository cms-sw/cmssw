#ifndef __DQMOffline_PFTau_MatchMETBenchmarkAnalyzer__
#define __DQMOffline_PFTau_MatchMETBenchmarkAnalyzer__

#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/MatchMETBenchmark.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

class TH1F; 

class MatchMETBenchmarkAnalyzer: public BenchmarkAnalyzer, public MatchMETBenchmark {
 public:
  
  MatchMETBenchmarkAnalyzer(const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob() ;
  void endJob();

 protected:
  edm::EDGetTokenT< edm::View<reco::MET> > myColl_;
  edm::EDGetTokenT< edm::View<reco::MET> > myMatchColl_;
  edm::InputTag matchedInputLabel_;
};

#endif 
