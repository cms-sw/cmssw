#include "DataFormats/SiStripCluster/interface/SiStripClustersSOA.h"

SiStripClustersSOA::SiStripClustersSOA(size_t maxClusters, int clustersPerStrip) {
  clusterIndex_ = std::make_unique<uint32_t[]>(maxClusters);
  clusterSize_ = std::make_unique<uint32_t[]>(maxClusters);
  clusterADCs_ = std::make_unique<uint8_t[]>(maxClusters * clustersPerStrip);
  clusterDetId_ = std::make_unique<stripgpu::detId_t[]>(maxClusters);
  firstStrip_ = std::make_unique<stripgpu::stripId_t[]>(maxClusters);
  trueCluster_ = std::make_unique<bool[]>(maxClusters);
  barycenter_ = std::make_unique<float[]>(maxClusters);
  charge_ = std::make_unique<float[]>(maxClusters);
}
