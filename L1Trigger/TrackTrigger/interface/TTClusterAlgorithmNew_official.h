#ifndef L1Trigger_TrackTrigger_TTClusterAlgorithmNew_official_h
#define L1Trigger_TrackTrigger_TTClusterAlgorithmNew_official_h

/*! \class   TTClusterAlgorithmNew_official
 *  \brief   Class for "official" algorithm to be used
 *           in TTClusterBuilderNew
 *  \details This implements the algorithm in the FE electronics.
 *           Clustering is done in 1D (in r-phi = rows).
 *           Clusters exceeding a configured width are removed.
 *
 *           For PS-p sensors only, the algo has additional rules:
 *             a) Clusters are split at MPA chip boundaries 
 *                (120 strip multiples), as MPA has no lateral communication.
 *             b) If two PS-p clusters in neighbouring columns (r-z) have 
 *                the same row (r-phi) centroid, the one with higher column
 *                number is vetoed.
 *             c) If >= 3 PS-p clusters in neighbouring columns (r-z) have 
 *                the same row (r-phi) centroid, all these clusters are 
 *                vetoed.
 *     
 *           (The code is based on that of the offline cluster producer
 *           Phase2TrackerCluserizer).
 *
 *  \author Ian Tomalin
 *  \date Aug. 2026
 */


#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

class TTClusterAlgorithmNew_official {
public:
  
  TTClusterAlgorithmNew_official(unsigned int maxClusterWidth, bool enableClusterVetoes, const DetId& detId, bool upperSensor, bool isPSp) : maxClusterWidth_(maxClusterWidth), enableClusterVetoes_(enableClusterVetoes), detId_(detId), upperSensor_(upperSensor), isPSp_(isPSp) {} 

  // Top-level of clustering algo
  void clusterizeDetUnit(const edm::DetSet<Phase2TrackerDigi>& digis,
                         TTClusterDetSetVec::FastFiller& clusters) const;

private:

  // Algo implmentation WITHOUT PS-p specific cluster vetoes
  void algo(const edm::DetSet<Phase2TrackerDigi>& digis,
            TTClusterDetSetVec::FastFiller& clusters) const;

  // Algo implmentation WITH PS-p specific cluster vetoes
  void algoWithVetoes(const edm::DetSet<Phase2TrackerDigi>& digis,
            TTClusterDetSetVec::FastFiller& clusters) const;

private:
  // Clustering algo cfg
  unsigned int maxClusterWidth_;
  bool enableClusterVetoes_;
  
  // Info about this tracker module
  DetId detId_;
  bool upperSensor_;
  bool isPSp_;
};

#endif
