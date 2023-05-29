#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

GEMInternalCluster::GEMInternalCluster(const GEMDetId& id1,
                                       const GEMDetId& id2,
                                       const GEMPadDigiCluster& cluster1,
                                       const GEMPadDigiCluster& cluster2,
                                       const unsigned delayGEMinOTMB,
                                       const unsigned tmbL1aWindowSize) {
  // set coincidence to false first
  isCoincidence_ = false;
  isValid_ = false;

  // set matches to false first
  isMatchingLayer1_ = false;
  isMatchingLayer2_ = false;

  id1_ = id1;
  id2_ = id2;

  if (cluster1.isValid()) {
    isValid_ = true;
    cl1_ = cluster1;
    if (cluster1.alctMatchTime() == -1)  // It is a MC simulation
      bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX;
    else if (cluster1.alctMatchTime() >= 0)  // It is real data
      bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX - int(tmbL1aWindowSize / 2) - cluster1.alctMatchTime() +
            delayGEMinOTMB;
    layer1_pad_ = cluster1.pads()[0];
    layer1_size_ = cluster1.pads().size();
  }
  if (cluster2.isValid()) {
    isValid_ = true;
    cl2_ = cluster2;
    if (cluster2.alctMatchTime() == -1)  // It is a MC simulation
      bx_ = cluster2.bx() + CSCConstants::LCT_CENTRAL_BX;
    else if (cluster2.alctMatchTime() >= 0)  // It is real data
      bx_ = cluster2.bx() + CSCConstants::LCT_CENTRAL_BX - int(tmbL1aWindowSize / 2) - cluster2.alctMatchTime() +
            delayGEMinOTMB;
    layer2_pad_ = cluster2.pads()[0];
    layer2_size_ = cluster2.pads().size();
  }

  if (cluster1.isValid() and cluster2.isValid()) {
    if (cluster1.alctMatchTime() == -1)  // It is a MC simulation
      bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX;
    else if (cluster1.alctMatchTime() >= 0)  // It is real data
      bx_ = cluster1.bx() + CSCConstants::LCT_CENTRAL_BX - int(tmbL1aWindowSize / 2) - cluster1.alctMatchTime() +
            delayGEMinOTMB;
    isCoincidence_ = true;
  }

  layer1_min_wg_ = -1;
  layer1_max_wg_ = -1;
  layer2_min_wg_ = -1;
  layer2_max_wg_ = -1;

  layer1_first_es_ = -1;
  layer2_first_es_ = -1;
  layer1_last_es_ = -1;
  layer2_last_es_ = -1;

  layer1_first_es_me1a_ = -1;
  layer2_first_es_me1a_ = -1;
  layer1_last_es_me1a_ = -1;
  layer2_last_es_me1a_ = -1;

  layer1_middle_es_ = -1;
  layer2_middle_es_ = -1;

  layer1_middle_es_me1a_ = -1;
  layer2_middle_es_me1a_ = -1;
}

GEMInternalCluster::GEMInternalCluster() {
  // set coincidence to false first
  isCoincidence_ = false;
  isValid_ = false;

  // set matches to false first
  isMatchingLayer1_ = false;
  isMatchingLayer2_ = false;

  layer1_min_wg_ = -1;
  layer1_max_wg_ = -1;
  layer2_min_wg_ = -1;
  layer2_max_wg_ = -1;

  layer1_first_es_ = -1;
  layer2_first_es_ = -1;
  layer1_last_es_ = -1;
  layer2_last_es_ = -1;

  layer1_first_es_me1a_ = -1;
  layer2_first_es_me1a_ = -1;
  layer1_last_es_me1a_ = -1;
  layer2_last_es_me1a_ = -1;

  layer1_middle_es_ = -1;
  layer2_middle_es_ = -1;

  layer1_middle_es_me1a_ = -1;
  layer2_middle_es_me1a_ = -1;
}

GEMPadDigi GEMInternalCluster::mid1() const {
  if (!cl1_.isValid())
    return GEMPadDigi();
  const unsigned pad = cl1_.pads()[cl1_.pads().size() / 2];

  return GEMPadDigi(pad, cl1_.bx(), cl1_.station(), cl1_.nPartitions());
}

GEMPadDigi GEMInternalCluster::mid2() const {
  if (!cl2_.isValid())
    return GEMPadDigi();
  const unsigned pad = cl2_.pads()[cl2_.pads().size() / 2];

  return GEMPadDigi(pad, cl2_.bx(), cl2_.station(), cl2_.nPartitions());
}

uint16_t GEMInternalCluster::getKeyStrip(int n, bool isLayer2) const {
  if (n == 8) {
    if (!isLayer2) {
      return (layer1_first_es_ + layer1_last_es_) / 2.;
    } else {
      return (layer2_first_es_ + layer2_last_es_) / 2.;
    }
  } else {  // Half Strip units
    if (!isLayer2) {
      return (layer1_first_es_ + layer1_last_es_) / 8.;
    } else {
      return (layer2_first_es_ + layer2_last_es_) / 8.;
    }
  }
}

uint16_t GEMInternalCluster::getKeyStripME1a(int n, bool isLayer2) const {
  if (n == 8) {
    if (!isLayer2) {
      return (layer1_first_es_me1a_ + layer1_last_es_me1a_) / 2.;
    } else {
      return (layer2_first_es_me1a_ + layer2_last_es_me1a_) / 2.;
    }
  } else {  // Half Strip units
    if (!isLayer2) {
      return (layer1_first_es_me1a_ + layer1_last_es_me1a_) / 8.;
    } else {
      return (layer2_first_es_me1a_ + layer2_last_es_me1a_) / 8.;
    }
  }
}

bool GEMInternalCluster::has_cluster(const GEMPadDigiCluster& cluster) const {
  return cl1_ == cluster or cl2_ == cluster;
}

bool GEMInternalCluster::operator==(const GEMInternalCluster& cluster) const {
  return id1_ == cluster.id1() and id2_ == cluster.id2() and cl1_ == cluster.cl1() and cl2_ == cluster.cl2();
}

std::ostream& operator<<(std::ostream& os, const GEMInternalCluster& cl) {
  return os << "Cluster Layer 1: " << cl.id1() << " " << cl.cl1() << ", Cluster Layer 2: " << cl.id2() << " "
            << cl.cl2();
}
