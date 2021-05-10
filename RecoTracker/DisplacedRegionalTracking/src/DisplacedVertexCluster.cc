#include "RecoTracker/DisplacedRegionalTracking/interface/DisplacedVertexCluster.h"

DisplacedVertexCluster::DisplacedVertexCluster(const edm::View<reco::VertexCompositeCandidate> &trackClusters,
                                               const unsigned index,
                                               const double rParam)
    : valid_(true),
      rParam2_(rParam * rParam),
      sumOfCenters_(trackClusters.at(index).vertex()),
      centerOfMass_(trackClusters.at(index).vertex()) {
  const auto trackClusterPtr = &trackClusters.at(index);
  constituents_.push_back(trackClusterPtr);
}

void DisplacedVertexCluster::merge(const DisplacedVertexCluster &other) {
  for (const auto &trackCluster : other.constituents())
    constituents_.push_back(trackCluster);
  sumOfCenters_ += other.sumOfCenters();
  centerOfMass_ = sumOfCenters_ * (1.0 / constituents_.size());
}

double DisplacedVertexCluster::Distance::distance2() const {
  if (entities_.first->valid() && entities_.second->valid())
    return (entities_.first->centerOfMass() - entities_.second->centerOfMass()).mag2();
  return std::numeric_limits<double>::max();
}
