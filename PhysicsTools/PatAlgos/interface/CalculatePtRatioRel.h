#ifndef __PhysicsTools_PatAlgos_CalculatePtRatioRel__
#define __PhysicsTools_PatAlgos_CalculatePtRatioRel__

#include "DataFormats/BTauReco/interface/JetTag.h"

#include <memory>
#include <string>

namespace pat {
  class Muon;
}

namespace reco {
  class JetCorrector;
}  // namespace reco

namespace pat {
  class CalculatePtRatioRel {
  public:
    CalculatePtRatioRel(float dR2max);

    ~CalculatePtRatioRel();

    std::array<float, 2> computePtRatioRel(const pat::Muon& imuon,
                                           const reco::JetTagCollection& bTags,
                                           const reco::JetCorrector* correctorL1 = nullptr,
                                           const reco::JetCorrector* correctorL1L2L3Res = nullptr) const;

  private:
    float dR2max_;
  };
}  // namespace pat
#endif
