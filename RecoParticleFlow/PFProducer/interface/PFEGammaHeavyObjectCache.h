#ifndef __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
#define __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include <memory>

namespace pfEGHelpers {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet& conf) {
      gbrEle_ = std::make_unique<GBRForest>(conf.getParameter<edm::FileInPath>("pf_electronID_mvaWeightFile"));
      gbrSingleLeg_ = std::make_unique<GBRForest>(conf.getParameter<edm::FileInPath>("pf_convID_mvaWeightFile"));
  }

    std::unique_ptr<const GBRForest> gbrEle_;
    std::unique_ptr<const GBRForest> gbrSingleLeg_;
  };
}

#endif // __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
