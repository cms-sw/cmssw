#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"

SiStripApproximateCluster::SiStripApproximateCluster(const SiStripCluster& cluster) {
  barycenter_ = cluster.barycenter();
  width_ = cluster.size();
  avgCharge_ = cluster.charge() / cluster.size();
}
