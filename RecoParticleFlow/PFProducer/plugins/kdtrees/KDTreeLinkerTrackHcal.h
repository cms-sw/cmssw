#ifndef KDTreeLinkerTrackHcal_h
#define KDTreeLinkerTrackHcal_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"


// This class is used to find all links between Tracks and HCAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerTrackHcal : public KDTreeLinkerBase
{
 public:
  KDTreeLinkerTrackHcal();
  ~KDTreeLinkerTrackHcal();
  
  // With this method, we create the list of psCluster that we want to link.
  void insertTargetElt(reco::PFBlockElement		*track);

  // Here, we create the list of hcalCluster that we want to link. From hcalCluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  void insertFieldClusterElt(reco::PFBlockElement	*hcalCluster);  

  // The KDTree building from rechits list.
  void buildTree();
  
  // Here we will iterate over all tracks. For each track intersection point with HCAL, 
  // we will search the closest rechits in the KDTree, from rechits we will find the 
  // hcalClusters and after that we will check the links between the track and 
  // all closest hcalClusters.  
  void searchLinks();
    
  // Here, we will store all PS/HCAL founded links in the PFBlockElement class
  // of each psCluster in the PFmultilinks field.
  void updatePFBlockEltWithLinks();
  
  // Here we free all allocated structures.
  void clear();
 
 private:
  // Data used by the KDTree algorithm : sets of Tracks and HCAL clusters.
  BlockEltSet		targetSet_;
  BlockEltSet		fieldClusterSet_;

  // Sets of rechits that compose the HCAL clusters. 
  RecHitSet		rechitsSet_;
  
  // Map of linked Track/HCAL clusters.
  BlockElt2BlockEltMap	cluster2TargetLinks_;

  // Map of the HCAL clusters associated to a rechit.
  RecHit2BlockEltMap	rechit2ClusterLinks_;
    
  // KD trees
  KDTreeLinkerAlgo	tree_;

};

#endif /* !KDTreeLinkerTrackHcal_h */
