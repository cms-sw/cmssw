#ifndef FastSimulation_TrackingRecHitProducer_FastPixelCPE_H
#define FastSimulation_TrackingRecHitProducer_FastPixelCPE_H

//Header files
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"


class FastPixelCPE final : public PixelClusterParameterEstimator
{
 public:
  FastPixelCPE(){;}

  // these methods should retrieve the information from the SiPixelCluster itself
  virtual ReturnType getParameters(const SiPixelCluster & cl,
                                   const GeomDetUnit    & det ) const override {
    return std::make_tuple(LocalPoint(),LocalError(),SiPixelRecHitQuality::QualWordType());
  }
  virtual ReturnType getParameters(const SiPixelCluster & cl,
                                   const GeomDetUnit    & det,
                                   const LocalTrajectoryParameters & ltp ) const override {
    return std::make_tuple(LocalPoint(),LocalError(),SiPixelRecHitQuality::QualWordType());
  }

  
};

#endif




