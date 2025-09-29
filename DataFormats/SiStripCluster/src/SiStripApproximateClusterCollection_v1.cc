#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection_v1.h"
using namespace v1;
void SiStripApproximateClusterCollection::reserve(std::size_t dets, std::size_t clusters) {
  detIds_.reserve(dets);
  clusters_.reserve(clusters);
}

SiStripApproximateClusterCollection::Filler SiStripApproximateClusterCollection::beginDet(unsigned int detId) {
  detIds_.push_back((detIds_.empty()) ? detId : detId - (std::accumulate(detIds_.cbegin(), detIds_.cend(), 0)));
  return Filler(clusters_);
}
