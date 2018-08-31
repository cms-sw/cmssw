#include "RecoParticleFlow/PFTracking/interface/ConvBremHeavyObjectCache.h"

namespace convbremhelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {

    pfcalib_ = std::make_unique<PFEnergyCalibration>();

    const bool useConvBremFinder_ = conf.getParameter<bool>("useConvBremFinder");

    if(useConvBremFinder_) {

      gbrBarrelLowPt_   = std::make_unique<const GBRForest>(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelLowPt"));
      gbrBarrelHighPt_  = std::make_unique<const GBRForest>(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelHighPt"));
      gbrEndcapsLowPt_  = std::make_unique<const GBRForest>(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsLowPt"));
      gbrEndcapsHighPt_ = std::make_unique<const GBRForest>(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsHighPt"));

    }
  }
}
