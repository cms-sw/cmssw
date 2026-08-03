#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerSequentialAlgorithm_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerSequentialAlgorithm_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

class Phase2TrackerClusterizerSequentialAlgorithm {
public:
  // maxClusterSize is the front end limit on the number of channels the readout puts in a
  // single cluster: a longer run of hit channels comes out as several adjacent clusters,
  // the first ones exactly maxClusterSize long. 0 means no limit.
  explicit Phase2TrackerClusterizerSequentialAlgorithm(unsigned int maxClusterSize = 0)
      : maxClusterSize_(maxClusterSize) {}

  inline void clusterizeDetUnit(const edm::DetSet<Phase2TrackerDigi>&,
                                Phase2TrackerCluster1DCollectionNew::FastFiller&) const;

private:
  unsigned int maxClusterSize_;
};

void Phase2TrackerClusterizerSequentialAlgorithm::clusterizeDetUnit(
    const edm::DetSet<Phase2TrackerDigi>& digis, Phase2TrackerCluster1DCollectionNew::FastFiller& clusters) const {
  if (digis.empty())
    return;
  auto di = digis.begin();
  unsigned int sizeCluster = 1;
  Phase2TrackerDigi firstDigi = *di;
  bool HIPbit = firstDigi.overThreshold();
  auto previous = firstDigi;
  ++di;
  for (; di != digis.end(); ++di) {
    auto digi = *di;
#ifdef VERIFY_PH2_TK_CLUS
    if (!(previous < digi))
      std::cout << "not ordered " << previous << ' ' << digi << std::endl;
#endif
    // Once the open cluster has reached the front end limit the digi cannot go into it,
    // so it opens the next one even though the two are adjacent.
    if (digi - previous == 1 and (maxClusterSize_ == 0 or sizeCluster < maxClusterSize_)) {
      HIPbit |= digi.overThreshold();
      ++sizeCluster;
    } else {
      clusters.push_back(Phase2TrackerCluster1D(firstDigi, sizeCluster, HIPbit));
      firstDigi = digi;
      HIPbit = digi.overThreshold();
      sizeCluster = 1;
    }
    previous = digi;
  }
  clusters.push_back(Phase2TrackerCluster1D(firstDigi, sizeCluster, HIPbit));
}

#endif
