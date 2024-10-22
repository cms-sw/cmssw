#ifndef L1Trigger_CSCTriggerPrimitives_GEMInternalCluster_h
#define L1Trigger_CSCTriggerPrimitives_GEMInternalCluster_h

/** \class GEMInternalCluster
 *
 * Helper class to contain detids, clusters and corresponding
 * 1/2-strips, 1/8-strips and wiregroups for easy matching with CSC TPs
 *
 * Author: Sven Dildick (Rice University)
 * Updates by Giovanni Mocellin (UC Davis)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"

class GEMInternalCluster {
public:
  // constructor
  GEMInternalCluster(const GEMDetId& id1,
                     const GEMDetId& id2,
                     const GEMPadDigiCluster& cluster1,
                     const GEMPadDigiCluster& cluster2,
                     const unsigned delayGEMinOTMB,
                     const unsigned tmbL1aWindowSize);

  // empty object
  GEMInternalCluster();

  GEMDetId id1() const { return id1_; }
  GEMDetId id2() const { return id2_; }
  GEMPadDigiCluster cl1() const { return cl1_; }
  GEMPadDigiCluster cl2() const { return cl2_; }
  bool isMatchingLayer1() const { return isMatchingLayer1_; }
  bool isMatchingLayer2() const { return isMatchingLayer2_; }

  // setter for coincidence
  void set_coincidence(const bool isCoincidence) { isCoincidence_ = isCoincidence; }

  // setter for detIDs
  void set_matchingLayer1(const bool isMatching) { isMatchingLayer1_ = isMatching; }
  void set_matchingLayer2(const bool isMatching) { isMatchingLayer2_ = isMatching; }

  // an internal cluster is valid if at least one is valid
  bool isValid() const { return isValid_; }

  // return the centers of the pads
  GEMPadDigi mid1() const;
  GEMPadDigi mid2() const;

  // return the coincidence pad
  GEMCoPadDigi copad() const;

  int bx() const { return bx_; }
  int roll1() const { return id1_.roll(); }
  int roll2() const { return id2_.roll(); }
  int layer1_pad() const { return layer1_pad_; }
  int layer1_size() const { return layer1_size_; }
  int layer2_pad() const { return layer2_pad_; }
  int layer2_size() const { return layer2_size_; }
  int layer1_min_wg() const { return layer1_min_wg_; }
  int layer1_max_wg() const { return layer1_max_wg_; }
  int layer2_min_wg() const { return layer2_min_wg_; }
  int layer2_max_wg() const { return layer2_max_wg_; }
  bool isCoincidence() const { return isCoincidence_; }

  // return "key wiregroup" and "key half-strip" for a cluster
  // these are approximate numbers obviously for LCTs with lower quality
  unsigned getKeyWG() const { return (layer2_min_wg() + layer2_max_wg()) / 2.; }
  uint16_t getKeyStrip(int n = 2, bool isLayer2 = false) const;
  uint16_t getKeyStripME1a(int n = 2, bool isLayer2 = false) const;

  // first and last 1/8-strips
  int layer1_first_es() const { return layer1_first_es_; }
  int layer2_first_es() const { return layer2_first_es_; }
  int layer1_last_es() const { return layer1_last_es_; }
  int layer2_last_es() const { return layer2_last_es_; }

  int layer1_first_es_me1a() const { return layer1_first_es_me1a_; }
  int layer2_first_es_me1a() const { return layer2_first_es_me1a_; }
  int layer1_last_es_me1a() const { return layer1_last_es_me1a_; }
  int layer2_last_es_me1a() const { return layer2_last_es_me1a_; }

  // middle 1/8-strips (sum divided by two)
  int layer1_middle_es() const { return layer1_middle_es_; }
  int layer2_middle_es() const { return layer2_middle_es_; }

  int layer1_middle_es_me1a() const { return layer1_middle_es_me1a_; }
  int layer2_middle_es_me1a() const { return layer2_middle_es_me1a_; }

  // setters for first/last 1/8-strip
  void set_layer1_first_es(const int es) { layer1_first_es_ = es; }
  void set_layer2_first_es(const int es) { layer2_first_es_ = es; }
  void set_layer1_last_es(const int es) { layer1_last_es_ = es; }
  void set_layer2_last_es(const int es) { layer2_last_es_ = es; }

  void set_layer1_first_es_me1a(const int es) { layer1_first_es_me1a_ = es; }
  void set_layer2_first_es_me1a(const int es) { layer2_first_es_me1a_ = es; }
  void set_layer1_last_es_me1a(const int es) { layer1_last_es_me1a_ = es; }
  void set_layer2_last_es_me1a(const int es) { layer2_last_es_me1a_ = es; }

  // setters for middle 1/8-strip
  void set_layer1_middle_es(const int es) { layer1_middle_es_ = es; }
  void set_layer2_middle_es(const int es) { layer2_middle_es_ = es; }
  void set_layer1_middle_es_me1a(const int es) { layer1_middle_es_me1a_ = es; }
  void set_layer2_middle_es_me1a(const int es) { layer2_middle_es_me1a_ = es; }

  // set the corresponding wiregroup numbers
  void set_layer1_min_wg(const int wg) { layer1_min_wg_ = wg; }
  void set_layer1_max_wg(const int wg) { layer1_max_wg_ = wg; }
  void set_layer2_min_wg(const int wg) { layer2_min_wg_ = wg; }
  void set_layer2_max_wg(const int wg) { layer2_max_wg_ = wg; }

  bool has_cluster(const GEMPadDigiCluster& cluster) const;

  // equality operator: detid, cluster 1 and cluster 2
  bool operator==(const GEMInternalCluster& cluster) const;

private:
  /*
    Detector id. For single clusters in layer 1 the GEMDetId in layer 1 is stored.
    Similarly, for single clusters in layer 2 the GEMDetId in layer 2 is stored.
    For coincidences the  GEMDetId both are stored.
  */
  GEMDetId id1_;
  GEMDetId id2_;
  GEMPadDigiCluster cl1_;
  GEMPadDigiCluster cl2_;

  // set matches to false first
  bool isMatchingLayer1_;
  bool isMatchingLayer2_;

  bool isValid_;

  // bunch crossing
  int bx_;

  // starting pads and sizes of the clusters in each layer
  // depending on the presence of a coincidence, layer 1, layer2, or both
  // can be filled
  int layer1_pad_;
  int layer1_size_;
  int layer2_pad_;
  int layer2_size_;

  // corresponding CSC 1/8-strip coordinates (es) of the cluster
  // in each layer (if applicable)
  int layer1_first_es_;
  int layer1_last_es_;
  int layer2_first_es_;
  int layer2_last_es_;
  // for ME1/a
  int layer1_first_es_me1a_;
  int layer1_last_es_me1a_;
  int layer2_first_es_me1a_;
  int layer2_last_es_me1a_;

  // middle CSC 1/8-strip
  int layer1_middle_es_;
  int layer2_middle_es_;
  // for ME1/a
  int layer1_middle_es_me1a_;
  int layer2_middle_es_me1a_;

  // corresponding min and max wiregroup
  int layer1_min_wg_;
  int layer1_max_wg_;
  int layer2_min_wg_;
  int layer2_max_wg_;

  // flag to signal if it is a coincidence
  bool isCoincidence_;
};

std::ostream& operator<<(std::ostream& os, const GEMInternalCluster& cl);

#endif
