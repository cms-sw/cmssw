// A minimal test to ensure that
//   - sistrip::SiStripClustersSoA, sistrip::SiStripClustersHost can be compiled
//   - sistrip::SiStripClustersSoA can be allocated, modified and erased (on host)
//   - view-based element access works

#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersHost.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"

int main() {
  constexpr const int kMaxSeedStrips = 200000;

  sistrip::SiStripClustersHost collection(kMaxSeedStrips, cms::alpakatools::host());
  collection.zeroInitialise();
  auto clust_view = collection.view();
  for (int j = 0; j < kMaxSeedStrips; ++j) {
    clust_view.clusterIndex(j) = (uint32_t)(j);
    clust_view.clusterSize(j) = (uint32_t)(j * 2);
    for (int k = 0; k < sistrip::maxStripsPerCluster; ++k) {
      clust_view.clusterADCs(j)[k] = (uint8_t)((j + k) % 255);
    }
    clust_view.clusterDetId(j) = (uint32_t)(j + 12);
    clust_view.firstStrip(j) = (uint16_t)(j % 65536);
    clust_view.trueCluster(j) = (bool)((j % 2 == 0));
    clust_view.barycenter(j) = (float)(j * 1.0f);
    clust_view.charge(j) = (float)(j * -1.0f);
  }
  clust_view.nClusters() = kMaxSeedStrips;
  clust_view.maxClusterSize() = sistrip::maxStripsPerCluster;

  // Assert
  for (int j = 0; j < kMaxSeedStrips; ++j) {
    assert(clust_view.clusterIndex(j) == (uint32_t)(j));
    assert(clust_view.clusterSize(j) == (uint32_t)(j * 2));
    for (int k = 0; k < sistrip::maxStripsPerCluster; ++k) {
      assert(clust_view.clusterADCs(j)[k] == (uint8_t)((j + k) % 255));
    }
    assert(clust_view.clusterDetId(j) == (uint32_t)(j + 12));
    assert(clust_view.firstStrip(j) == (uint16_t)(j % 65536));
    assert(clust_view.trueCluster(j) == (bool)((j % 2 == 0)));
    assert(clust_view.barycenter(j) == (float)(j * 1.0f));
    assert(clust_view.charge(j) == (float)(j * -1.0f));
  }
  assert(clust_view.nClusters() == kMaxSeedStrips);
  assert(clust_view.maxClusterSize() == sistrip::maxStripsPerCluster);

  return 0;
}
