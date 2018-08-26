#include "RecoParticleFlow/PFTracking/interface/ConvBremHeavyObjectCache.h"

namespace convbremhelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {

    pfcalib_.reset( new PFEnergyCalibration() );

    const bool useConvBremFinder_ = conf.getParameter<bool>("useConvBremFinder");

    if(useConvBremFinder_) {

      gbrBarrelLowPt_   = setupMVA(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelLowPt"));
      gbrBarrelHighPt_  = setupMVA(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelHighPt"));
      gbrEndcapsLowPt_  = setupMVA(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsLowPt"));
      gbrEndcapsHighPt_ = setupMVA(conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsHighPt"));

    }
  }

  std::unique_ptr<const GBRForest> HeavyObjectCache::setupMVA(const std::string& weights) {
    return std::make_unique<const GBRForest>( weights );
  }  
}
