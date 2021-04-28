#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

GEMInternalCluster::GEMInternalCluster(const GEMDetId& id,
                                       const GEMPadDigiCluster& cluster1,
                                       const GEMPadDigiCluster& cluster2) {
  id_ = id;

  if (cluster1.isValid()) {
    cl1_ = cluster1;
    bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX;
    layer1_pad_ = cluster1.pads()[0];
    layer1_size_ = cluster1.pads().size();
  }
  if (cluster2.isValid()) {
    cl2_ = cluster2;
    bx_ = cluster2.bx() + CSCConstants::LCT_CENTRAL_BX;
    layer2_pad_ = cluster2.pads()[0];
    layer2_size_ = cluster2.pads().size();
  }

  if (cluster1.isValid() and cluster2.isValid()) {
    bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX;
    isCoincidence_ = true;
  }

  layer1_first_es_ = -1;
  layer1_last_es_ = -1;
  layer2_first_es_ = -1;
  layer2_last_es_ = -1;
  layer1_first_es_me1a_ = -1;
  layer1_last_es_me1a_ = -1;
  layer2_first_es_me1a_ = -1;
  layer2_last_es_me1a_ = -1;
  min_wg_ = -1;
  max_wg_ = -1;
}

bool GEMInternalCluster::has_cluster(const GEMPadDigiCluster& cluster) const {
  return cl1_ == cluster or cl2_ == cluster;
}

bool GEMInternalCluster::operator==(const GEMInternalCluster& cluster) const {
  return id_ == cluster.id() and cl1_ == cluster.cl1() and cl2_ == cluster.cl2();
}
