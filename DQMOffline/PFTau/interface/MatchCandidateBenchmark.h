#ifndef RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h
#define RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>

/// To plot Candidate quantities
class MatchCandidateBenchmark : public Benchmark {

 public:

  MatchCandidateBenchmark(Mode mode) : Benchmark(mode) {}
  virtual ~MatchCandidateBenchmark();

  /// book histograms
  void setup();
  
  /// fill histograms with a given particle
  void fillOne( const reco::Candidate& candidate,
		const reco::Candidate& matchedCandidate ); 


 protected:
  
  TH2F*   delta_et_Over_et_VS_et_; 
  TH2F*   delta_et_VS_et_; 

};

 

#endif 
