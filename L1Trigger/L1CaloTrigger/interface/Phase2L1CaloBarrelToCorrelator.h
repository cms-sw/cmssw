#ifndef PHASE_2_L1_CALO_BARREL_TO_CORRELATOR
#define PHASE_2_L1_CALO_BARREL_TO_CORRELATOR

#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h"

/* 
 * Comparators for sorting EG clusters
 */
inline bool p2eg::compareGCTEmDigiClusterET(const l1tp2::GCTEmDigiCluster& lhs, const l1tp2::GCTEmDigiCluster& rhs) {
  return (lhs.ptFloat() > rhs.ptFloat());
}

/*
 * Returns the difference in the azimuth coordinates of phi1 and phi2 (all in degrees not radians), taking the wrap-around at 180 degrees into account
 */
inline float p2eg::deltaPhiInDegrees(float phi1, float phi2, const float c = 180) {
  float r = std::fmod(phi1 - phi2, 2.0 * c);
  if (r < -c) {
    r += 2.0 * c;
  } else if (r > c) {
    r -= 2.0 * c;
  }
  return r;
}

/*
 * For a given phi in degrees (e.g. computed from some difference), return the phi (in degrees) which takes the wrap-around at 180 degrees into account
 */
inline float p2eg::wrappedPhiInDegrees(float phi) { return p2eg::deltaPhiInDegrees(phi, 0); }

/*
 * Sort the clusters in each egamma SLR in descending pT, then pad any zero clusters so that the total number of clusters in the SLR is six
 */
inline void p2eg::sortAndPad_eg_SLR(l1tp2::GCTEmDigiClusterLink& thisSLR) {
  // input is a vector and can be sorted
  std::sort(thisSLR.begin(), thisSLR.end(), p2eg::compareGCTEmDigiClusterET);
  int nClusters = thisSLR.size();

  // If there are fewer than the designated number of clusters, pad with zeros
  if (nClusters < p2eg::N_EG_CLUSTERS_PER_RCT_CARD) {
    // do padding. if size == 2, push back four clusters
    for (int i = 0; i < (p2eg::N_EG_CLUSTERS_PER_RCT_CARD - nClusters); i++) {
      l1tp2::GCTEmDigiCluster zeroCluster;
      thisSLR.push_back(zeroCluster);
    }
  }
  // If there are more than the designated number of clusters, truncate the vector
  else if (nClusters > p2eg::N_EG_CLUSTERS_PER_RCT_CARD) {
    // Get the iterator to the sixth element and delete til the end of the vector
    thisSLR.erase(thisSLR.begin() + p2eg::N_EG_CLUSTERS_PER_RCT_CARD, thisSLR.end());
  }
}

/* 
 * Comparators for sorting PF clusters
 */
inline bool p2eg::compareGCTHadDigiClusterET(const l1tp2::GCTHadDigiCluster& lhs, const l1tp2::GCTHadDigiCluster& rhs) {
  return (lhs.ptFloat() > rhs.ptFloat());
}

/*
 * Sort the clusters in each PF SLR in descending pT, then pad any zero clusters so that the total number of clusters in the SLR is six
 */
inline void p2eg::sortAndPad_had_SLR(l1tp2::GCTHadDigiClusterLink& thisSLR) {
  // input is a vector and can be sorted
  std::sort(thisSLR.begin(), thisSLR.end(), p2eg::compareGCTHadDigiClusterET);
  int nClusters = thisSLR.size();

  // If there are fewer than the designated number of clusters, pad with zeros
  if (nClusters < p2eg::N_PF_CLUSTERS_PER_RCT_CARD) {
    for (int i = 0; i < (p2eg::N_PF_CLUSTERS_PER_RCT_CARD - nClusters); i++) {
      l1tp2::GCTHadDigiCluster zeroCluster;
      thisSLR.push_back(zeroCluster);
    }
  }
  // If there are more than the designated number of clusters, truncate the vector
  else if (nClusters > p2eg::N_EG_CLUSTERS_PER_RCT_CARD) {
    // Get the iterator to the sixth element and delete til the end of the vector
    thisSLR.erase(thisSLR.begin() + p2eg::N_EG_CLUSTERS_PER_RCT_CARD, thisSLR.end());
  }
}

#endif