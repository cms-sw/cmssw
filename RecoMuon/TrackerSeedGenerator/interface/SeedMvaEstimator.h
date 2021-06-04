#ifndef SeedMvaEstimator_h
#define SeedMvaEstimator_h

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <memory>
#include <string>

class GBRForest;

namespace edm {
  class FileInPath;
}

class SeedMvaEstimator {
public:
  SeedMvaEstimator(const edm::FileInPath& weightsfile, std::vector<double> scale_mean, std::vector<double> scale_std);
  ~SeedMvaEstimator();

  std::vector<double> scale_mean_;
  std::vector<double> scale_std_;

  double computeMva(const TrajectorySeed&,
                    GlobalVector,
                    edm::Handle<l1t::MuonBxCollection>,
                    int minL1Qual,
                    edm::Handle<reco::RecoChargedCandidateCollection>,
                    bool isFromL1) const;

private:
  std::unique_ptr<const GBRForest> gbrForest_;

  void getL1MuonVariables(
      const TrajectorySeed&, GlobalVector, edm::Handle<l1t::MuonBxCollection>, int minL1Qual, float&, float&) const;
  void getL2MuonVariables(
      const TrajectorySeed&, GlobalVector, edm::Handle<reco::RecoChargedCandidateCollection>, float&, float&) const;
};
#endif
