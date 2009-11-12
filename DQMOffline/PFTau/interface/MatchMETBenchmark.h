#ifndef RecoParticleFlow_Benchmark_MatchMETBenchmark_h
#define RecoParticleFlow_Benchmark_MatchMETBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/METFwd.h"

// is this include necessary? 
// check all includes 

// integrate and check your benchmarks in PFRootEvent (take PFCandidateManager as an example)

// integrate and check your benchmarks Validation/RecoParticleFlow (take PFCandidateManager as an example)

// remove the old benchmarks from these 2 packages (python files, C++ code, ...)

#include <vector>

class MatchMETBenchmark : public Benchmark {

 public:

  MatchMETBenchmark(Mode mode) : Benchmark(mode) {}
  virtual ~MatchMETBenchmark();

  /// book histograms
  void setup();
  
  /// fill histograms with a given particle
  void fillOne( const reco::MET& candidate,
		const reco::MET& matchedCandidate ); 


 protected:
  // next 3: add to MatchCandidateBenchmark? 

  // (rec - true) / true = rec/true - 1 
  TH2F*   delta_et_VS_et_; 
  TH2F*   delta_et_Over_et_VS_et_; 

  TH2F*   delta_phi_VS_et_; 

  TH1F*   delta_ex_;

  // True and Rec: remove. remove the following histo? 
  TH2F*   RecEt_VS_TrueEt_;
  TH2F*   delta_set_VS_set_;
  TH2F*   delta_set_Over_set_VS_set_;
  TH2F*   delta_ex_VS_set_;

  // remove the following histo?
  TH2F*   RecSet_Over_TrueSet_VS_TrueSet_;

};

#endif 
