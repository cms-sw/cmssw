#ifndef RecoMuon_TrackerSeedGenerator_SeedMvaEstimator_h
#define RecoMuon_TrackerSeedGenerator_SeedMvaEstimator_h

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <memory>

class GBRForest;

namespace edm {
  class FileInPath;
}

namespace {
  enum inputIndexes {
    kTsosErr0,       // 0
    kTsosErr2,       // 1
    kTsosErr5,       // 2
    kTsosDxdz,       // 3
    kTsosDydz,       // 4
    kTsosQbp,        // 5
    kDRdRL1SeedP,    // 6
    kDPhidRL1SeedP,  // 7
    kLastL1,         // 8

    kDRdRL2SeedP = 8,  // 8
    kDPhidRL2SeedP,    // 9
    kLastL2,           // 10
  };
}  // namespace

class SeedMvaEstimator {
public:
  SeedMvaEstimator(const edm::FileInPath& weightsfile,
                   const std::vector<double>& scale_mean,
                   const std::vector<double>& scale_std,
                   const bool isFromL1,
                   const int minL1Qual);
  ~SeedMvaEstimator();

  double computeMva(const TrajectorySeed&,
                    const GlobalVector&,
                    const l1t::MuonBxCollection&,
                    const reco::RecoChargedCandidateCollection&) const;

private:
  std::unique_ptr<const GBRForest> gbrForest_;
  const std::vector<float> scale_mean_;
  std::vector<float> scale_istd_;
  const bool isFromL1_;
  const int minL1Qual_;

  void getL1MuonVariables(const GlobalVector&, const l1t::MuonBxCollection&, float&, float&) const;
  void getL2MuonVariables(const GlobalVector&, const reco::RecoChargedCandidateCollection&, float&, float&) const;
};
#endif
