#ifndef RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h
#define RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/CandidateBenchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <vector>

/// To plot Candidate quantities
class MatchCandidateBenchmark : public CandidateBenchmark {

 public:


  typedef reco::CandidateCollection Collection;

  MatchCandidateBenchmark();
  virtual ~MatchCandidateBenchmark();

  /// book histograms
  void setup();
  
  /// fill histograms with all particle
  void fill(const Collection& candCollection,
	    const Collection& matchedCandCollection );

 protected:
  
  /// fill histograms with a given particle
  void fill( const reco::Candidate& candidate,
	     const reco::Candidate& matchedCandidate ); 

  std::vector<int> match(const Collection& candCollection,
			 const Collection& matchedCandCollection ) {return std::vector<int>();}
 

  TH1F*   delta_pt_; 

};

#endif 
