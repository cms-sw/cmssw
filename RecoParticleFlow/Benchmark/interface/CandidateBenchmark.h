#ifndef RecoParticleFlow_Benchmark_CandidateBenchmark_h
#define RecoParticleFlow_Benchmark_CandidateBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/Benchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

/// To plot Candidate quantities
class CandidateBenchmark : public Benchmark {

 public:

  typedef reco::CandidateCollection Collection;

  CandidateBenchmark();
  virtual ~CandidateBenchmark();

  /// book histograms
  void setup();
  
  /// fill histograms with all particle
  void fill(const Collection& pfCandCollection );

 protected:
  
  /// fill histograms with a given particle
  void fill( const reco::Candidate& candidate); 

  TH1F*   pt_; 
  TH1F*   eta_; 
  TH1F*   phi_; 
  TH1F*   charge_; 

};

#endif 
