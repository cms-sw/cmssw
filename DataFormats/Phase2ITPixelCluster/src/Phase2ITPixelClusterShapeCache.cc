#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelClusterShapeCache.h"
#include "FWCore/Utilities/interface/Exception.h"

Phase2ITPixelClusterShapeData::~Phase2ITPixelClusterShapeData() {}

Phase2ITPixelClusterShapeCache::~Phase2ITPixelClusterShapeCache() {}

void Phase2ITPixelClusterShapeCache::checkRef(const ClusterRef& ref) const {
  if (ref.id() != productId_)
    throw cms::Exception("InvalidReference")
        << "Phase2ITPixelClusterShapeCache caches values for Phase2ITPixelClusters with ProductID " << productId_
        << ", got Phase2ITPixelClusterRef with ID " << ref.id();
  if (ref.index() >= data_.size())
    throw cms::Exception("InvalidReference")
        << "Phase2ITPixelClusterShapeCache caches values for Phase2ITPixelClusters with ProductID " << productId_
        << " that has " << data_.size() << " clusters, got Phase2ITPixelClusterRef with index " << ref.index();
}
