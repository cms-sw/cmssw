#include "RecoParticleFlow/PFTracking/interface/ConvBremHeavyObjectCache.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

namespace convbremhelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {

    pfcalib_ = std::make_unique<PFEnergyCalibration>();

    const bool useConvBremFinder_ = conf.getParameter<bool>("useConvBremFinder");

    if(useConvBremFinder_) {

      gbrBarrelLowPt_   = createGBRForest(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelLowPt"));
      gbrBarrelHighPt_  = createGBRForest(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelHighPt"));
      gbrEndcapsLowPt_  = createGBRForest(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsLowPt"));
      gbrEndcapsHighPt_ = createGBRForest(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsHighPt"));

    }
  }
}
