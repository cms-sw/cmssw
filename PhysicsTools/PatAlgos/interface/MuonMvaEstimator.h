#ifndef __PhysicsTools_PatAlgos_MuonMvaEstimator__
#define __PhysicsTools_PatAlgos_MuonMvaEstimator__

#include "DataFormats/BTauReco/interface/JetTag.h"

#include <memory>
#include <string>

class GBRForest;

namespace pat {
  class Muon;
}

namespace reco {
  class JetCorrector;
  class Vertex;
}  // namespace reco

namespace edm {
  class FileInPath;
}

namespace pat {
  class MuonMvaEstimator {
  public:
    MuonMvaEstimator(const edm::FileInPath& weightsfile, float dRmax);

    ~MuonMvaEstimator();

    float computeMva(const pat::Muon& imuon,
                     const reco::Vertex& vertex,
                     const reco::JetTagCollection& bTags,
                     float& jetPtRatio,
                     float& jetPtRel,
                     float& miniIsoValue,
                     const reco::JetCorrector* correctorL1 = nullptr,
                     const reco::JetCorrector* correctorL1L2L3Res = nullptr) const;

  private:
    std::unique_ptr<const GBRForest> gbrForest_;
    float dRmax_;
  };
}  // namespace pat
#endif
