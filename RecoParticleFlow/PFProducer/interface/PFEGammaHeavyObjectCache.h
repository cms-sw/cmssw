#ifndef __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
#define __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include <memory>

namespace pfEGHelpers {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet& conf)
      : gbrEle_       (createGBRForest(conf.getParameter<edm::FileInPath>("pf_electronID_mvaWeightFile")))
      , gbrSingleLeg_ (createGBRForest(conf.getParameter<edm::FileInPath>("pf_convID_mvaWeightFile")))
    {}

    const std::unique_ptr<const GBRForest> gbrEle_;
    const std::unique_ptr<const GBRForest> gbrSingleLeg_;
  };
}

#endif // __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
