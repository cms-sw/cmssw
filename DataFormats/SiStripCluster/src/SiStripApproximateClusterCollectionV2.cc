#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollectionV2.h"
void SiStripApproximateClusterCollectionV2::reserve(std::size_t dets, std::size_t clusters) {
  detIds_.reserve(dets);
  clusters_.reserve(clusters);
}

SiStripApproximateClusterCollectionV2::Filler SiStripApproximateClusterCollectionV2::beginDet(unsigned int detId) {
  detIds_.push_back((detIds_.size() == 0) ? detId : detId - (std::accumulate(detIds_.cbegin(), detIds_.cend(),0)));
  return Filler(clusters_);
}
