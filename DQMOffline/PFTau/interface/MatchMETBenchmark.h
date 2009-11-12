#ifndef RecoParticleFlow_Benchmark_MatchMETBenchmark_h
#define RecoParticleFlow_Benchmark_MatchMETBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/METFwd.h"

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
  
  TH2F*   delta_et_Over_et_VS_et_; 
  TH2F*   delta_et_VS_et_; 
  TH2F*   delta_phi_VS_et_; 
  TH1F*   delta_ex_;
  TH2F*   RecEt_VS_TrueEt_;
  TH2F*   delta_set_VS_set_;
  TH2F*   delta_set_Over_set_VS_set_;
  TH2F*   delta_ex_VS_set_;
  TH2F*   RecSet_Over_TrueSet_VS_TrueSet_;

};

#endif 
