#ifndef RecoParticleFlow_Benchmark_MatchMETBenchmark_h
#define RecoParticleFlow_Benchmark_MatchMETBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"

// integrate and check your benchmarks in PFRootEvent (take PFCandidateManager
// as an example)

// remove the old benchmarks from these 2 packages (Validation and PFRootEvent)
// (python files, C++ code, ...)
class MatchMETBenchmark : public Benchmark {
public:
  MatchMETBenchmark(Mode mode) : Benchmark(mode) {}
  ~MatchMETBenchmark() override;

  /// book histograms
  void setup(DQMStore::IBooker &b);

  /// fill histograms with a given particle
  void fillOne(const reco::MET &candidate, const reco::MET &matchedCandidate);

protected:
  // next 3: add to MatchCandidateBenchmark?

  TH2F *delta_et_VS_et_;
  TH2F *delta_et_Over_et_VS_et_;

  TH2F *delta_phi_VS_et_;

  TH1F *delta_ex_;

  // True and Rec: remove. remove the following histo?
  TH2F *RecEt_VS_TrueEt_;
  TH2F *delta_set_VS_set_;
  TH2F *delta_set_Over_set_VS_set_;
  TH2F *delta_ex_VS_set_;

  // remove the following histo?
  TH2F *RecSet_Over_TrueSet_VS_TrueSet_;
};

#endif
