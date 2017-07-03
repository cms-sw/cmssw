#ifndef KDTreeLinkerTrackEcal_h
#define KDTreeLinkerTrackEcal_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"


// This class is used to find all links between Tracks and ECAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerTrackEcal : public KDTreeLinkerBase
{
 public:
  KDTreeLinkerTrackEcal();
  ~KDTreeLinkerTrackEcal() override;
  
  // With this method, we create the list of psCluster that we want to link.
  void insertTargetElt(reco::PFBlockElement		*track) override;

  // Here, we create the list of ecalCluster that we want to link. From ecalCluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  void insertFieldClusterElt(reco::PFBlockElement	*ecalCluster) override;  

  // The KDTree building from rechits list.
  void buildTree() override;
  
  // Here we will iterate over all tracks. For each track intersection point with ECAL, 
  // we will search the closest rechits in the KDTree, from rechits we will find the 
  // ecalClusters and after that we will check the links between the track and 
  // all closest ecalClusters.  
  void searchLinks() override;
    
  // Here, we will store all PS/ECAL founded links in the PFBlockElement class
  // of each psCluster in the PFmultilinks field.
  void updatePFBlockEltWithLinks() override;
  
  // Here we free all allocated structures.
  void clear() override;
 
 private:
  // Data used by the KDTree algorithm : sets of Tracks and ECAL clusters.
  BlockEltSet		targetSet_;
  BlockEltSet		fieldClusterSet_;

  // Sets of rechits that compose the ECAL clusters. 
  RecHitSet		rechitsSet_;
  
  // Map of linked Track/ECAL clusters.
  BlockElt2BlockEltMap	target2ClusterLinks_;

  // Map of the ECAL clusters associated to a rechit.
  RecHit2BlockEltMap	rechit2ClusterLinks_;
    
  // KD trees
  KDTreeLinkerAlgo	tree_;

};

#endif /* !KDTreeLinkerTrackEcal_h */
