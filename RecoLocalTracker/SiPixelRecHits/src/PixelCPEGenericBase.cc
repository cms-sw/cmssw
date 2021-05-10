#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"

std::unique_ptr<PixelCPEBase::ClusterParam> PixelCPEGenericBase::createClusterParam(const SiPixelCluster& cl) const {
  return std::make_unique<ClusterParamGeneric>(cl);
}