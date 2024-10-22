#ifndef RecoParticleFlow_Benchmark_BenchmarkManager_h
#define RecoParticleFlow_Benchmark_BenchmarkManager_h

#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/PFCandidateBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <vector>

/// \brief A benchmark managing several benchmarks
///
/// The benchmarks are filled only for the PFCandidates matched to
/// a candidate in the matching collection within the limits in delta R.
/// The parameters for this benchmark are:
/// - the maximum delta R for matching
/// - the minimum pT of the reconstructed PFCandidate. Low pT PFCandidates have
/// to be removed, as they lead to a lot of hits in the histograms with delta
/// p_T just a bit larger than p_T_gen
/// - a bool, specifying if the charge of the candidates should be used in the
/// matching.
/// - the benchmark mode, driving the size of the histograms.
class PFCandidateManager : public Benchmark {
public:
  PFCandidateManager(float dRMax = 0.3, bool matchCharge = true, Benchmark::Mode mode = Benchmark::DEFAULT)
      : Benchmark(mode),
        candBench_(mode),
        pfCandBench_(mode),
        matchCandBench_(mode),
        dRMax_(dRMax),
        matchCharge_(matchCharge) {}

  ~PFCandidateManager() override;

  /// set the benchmark parameters
  void setParameters(float dRMax = 0.3, bool matchCharge = true, Benchmark::Mode mode = Benchmark::DEFAULT);

  /// set directory (to use in ROOT)
  void setDirectory(TDirectory *dir) override;

  /// book histograms
  void setup(DQMStore::IBooker &b);

  /// fill histograms with all particle
  template <class C>
  void fill(const reco::PFCandidateCollection &candCollection, const C &matchedCandCollection);

protected:
  CandidateBenchmark candBench_;
  PFCandidateBenchmark pfCandBench_;
  MatchCandidateBenchmark matchCandBench_;

  float dRMax_;
  bool matchCharge_;
};

#include "DQMOffline/PFTau/interface/Matchers.h"

template <class C>
void PFCandidateManager::fill(const reco::PFCandidateCollection &candCollection, const C &matchCandCollection) {
  std::vector<int> matchIndices;
  PFB::match(candCollection, matchCandCollection, matchIndices, matchCharge_, dRMax_);

  for (unsigned int i = 0; i < candCollection.size(); i++) {
    const reco::PFCandidate &cand = candCollection[i];

    if (!isInRange(cand.pt(), cand.eta(), cand.phi()))
      continue;

    int iMatch = matchIndices[i];

    assert(iMatch < static_cast<int>(matchCandCollection.size()));

    // COLIN how to handle efficiency plots?

    // filling the histograms in CandidateBenchmark only in case
    // of a matching.
    if (iMatch != -1) {
      candBench_.fillOne(cand);
      pfCandBench_.fillOne(cand);
      matchCandBench_.fillOne(cand, matchCandCollection[iMatch]);
    }
  }
}

#endif
