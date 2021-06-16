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

class SeedMvaEstimator {
public:
  SeedMvaEstimator(const edm::FileInPath& weightsfile,
                   const std::vector<double>& scale_mean,
                   const std::vector<double>& scale_std);
  ~SeedMvaEstimator();

  std::vector<double> scale_mean_;
  std::vector<double> scale_std_;

  double computeMva(const TrajectorySeed&,
                    const GlobalVector&,
                    const l1t::MuonBxCollection&,
                    int minL1Qual,
                    const reco::RecoChargedCandidateCollection&,
                    bool isFromL1) const;

private:
  std::unique_ptr<const GBRForest> gbrForest_;

  void getL1MuonVariables(const GlobalVector&, const l1t::MuonBxCollection&, int minL1Qual, float&, float&) const;
  void getL2MuonVariables(const GlobalVector&, const reco::RecoChargedCandidateCollection&, float&, float&) const;
};
#endif
