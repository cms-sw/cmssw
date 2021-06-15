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
  layer1_min_wg_ = -1;
  layer1_max_wg_ = -1;
  layer2_min_wg_ = -1;
  layer2_max_wg_ = -1;
}

GEMPadDigi GEMInternalCluster::mid1() const {
  if (cl1_.isValid())
    return GEMPadDigi();
  const unsigned pad = cl1_.pads()[cl1_.pads().size() / 2];

  return GEMPadDigi(pad, cl1_.bx(), cl1_.station(), cl1_.nPartitions());
}

GEMPadDigi GEMInternalCluster::mid2() const {
  if (cl2_.isValid())
    return GEMPadDigi();
  const unsigned pad = cl2_.pads()[cl2_.pads().size() / 2];

  return GEMPadDigi(pad, cl2_.bx(), cl2_.station(), cl2_.nPartitions());
}

int GEMInternalCluster::min_wg() const {
  if (id_.layer() == 1)
    return layer1_min_wg();
  else
    return layer2_min_wg();
}

int GEMInternalCluster::max_wg() const {
  if (id_.layer() == 1)
    return layer1_max_wg();
  else
    return layer2_max_wg();
}

uint16_t GEMInternalCluster::getKeyStrip(int n) const {
  // for ME2/1 and ME1/b return the average half-strip
  // in ME11: ME11
  // ME1b: keyWG >15,
  // ME1a and ME1b overlap:  10<=keyWG<=15
  // ME1a: keyWG < 10

  // case for half-strips
  if (n == 2) {
    // calculate the key wiregroup. If that is at least 10, go with ME1/b
    if (id_.station() == 2 or (id_.station() == 1 and getKeyWG() >= 10)) {
      if (id_.layer() == 1) {
        return (layer1_first_hs_ + layer1_last_hs_) / 2.;
      } else {
        return (layer2_first_hs_ + layer2_last_hs_) / 2.;
      }
    } else {
      if (id_.layer() == 1) {
        return (layer1_first_hs_me1a_ + layer1_last_hs_me1a_) / 2.;
      } else {
        return (layer2_first_hs_me1a_ + layer2_last_hs_me1a_) / 2.;
      }
    }
  }

  // case for 1/8-strips
  else {
    // calculate the key wiregroup. If that is at least 10, go with ME1/b
    if (id_.station() == 2 or (id_.station() == 1 and getKeyWG() >= 10)) {
      if (id_.layer() == 1) {
        return (layer1_first_es_ + layer1_last_es_) / 2.;
      } else {
        return (layer2_first_es_ + layer2_last_es_) / 2.;
      }
    } else {
      if (id_.layer() == 1) {
        return (layer1_first_es_me1a_ + layer1_last_es_me1a_) / 2.;
      } else {
        return (layer2_first_es_me1a_ + layer2_last_es_me1a_) / 2.;
      }
    }
  }
}

bool GEMInternalCluster::has_cluster(const GEMPadDigiCluster& cluster) const {
  return cl1_ == cluster or cl2_ == cluster;
}

bool GEMInternalCluster::operator==(const GEMInternalCluster& cluster) const {
  return id_ == cluster.id() and cl1_ == cluster.cl1() and cl2_ == cluster.cl2();
}
