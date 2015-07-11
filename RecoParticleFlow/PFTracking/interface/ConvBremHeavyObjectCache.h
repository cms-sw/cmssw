#ifndef __RecoParticleFlow_PFTracking_convbremhelpersHeavyObjectCache_h__
#define __RecoParticleFlow_PFTracking_convbremhelpersHeavyObjectCache_h__


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include <memory>

namespace convbremhelpers {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);
    std::unique_ptr<const GBRForest> gbrBarrelLowPt_;
    std::unique_ptr<const GBRForest> gbrBarrelHighPt_;
    std::unique_ptr<const GBRForest> gbrEndcapsLowPt_;
    std::unique_ptr<const GBRForest> gbrEndcapsHighPt_;
    std::unique_ptr<const PFEnergyCalibration> pfcalib_;
  private:
    std::unique_ptr<const GBRForest> setupMVA(const std::string&);
    // for variable binding
    float secR, sTIP, nHITS1, Epout, detaBremKF, ptRatioGsfKF;
  };
}

#endif
