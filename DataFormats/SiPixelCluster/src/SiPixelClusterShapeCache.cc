#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "FWCore/Utilities/interface/Exception.h"

SiPixelClusterShapeData::~SiPixelClusterShapeData() {}

SiPixelClusterShapeCache::LazyGetter::LazyGetter() {}
SiPixelClusterShapeCache::LazyGetter::~LazyGetter() {}

SiPixelClusterShapeCache::~SiPixelClusterShapeCache() {}

void SiPixelClusterShapeCache::checkRef(const ClusterRef& ref) const {
  if(ref.id() != productId_)
    throw cms::Exception("InvalidReference") << "SiPixelClusterShapeCache caches values for SiPixelClusters with ProductID " << productId_ << ", got SiPixelClusterRef with ID " << ref.id();
  if(ref.index() >= data_.size())
    throw cms::Exception("InvalidReference") << "SiPixelClusterShapeCache caches values for SiPixelClusters with ProductID " << productId_ << " that has " << data_.size() << " clusters, got SiPixelClusterRef with index " << ref.index();
}
