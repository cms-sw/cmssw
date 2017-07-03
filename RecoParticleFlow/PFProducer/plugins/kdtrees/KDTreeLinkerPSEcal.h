#ifndef KDTreeLinkerPSEcal_h
#define KDTreeLinkerPSEcal_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"


// This class is used to find all links between PreShower clusters and ECAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
class KDTreeLinkerPSEcal : public KDTreeLinkerBase
{
 public:
  KDTreeLinkerPSEcal();
  ~KDTreeLinkerPSEcal() override;
  
  // With this method, we create the list of psCluster that we want to link.
  void insertTargetElt(reco::PFBlockElement		*psCluster) override;

  // Here, we create the list of ecalCluster that we want to link. From ecalCluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  void insertFieldClusterElt(reco::PFBlockElement	*ecalCluster) override;  

  // The KDTree building from rechits list.
  void buildTree() override;
  
  // Here we will iterate over all psCluster. For each one, we will search the closest
  // rechits in the KDTree, from rechits we will find the ecalClusters and after that
  // we will check the links between the psCluster and all closest ecalClusters.
  void searchLinks() override;
  
  // Here, we will store all PS/ECAL founded links in the PFBlockElement class
  // of each psCluster in the PFmultilinks field.
  void updatePFBlockEltWithLinks() override;
  
  // Here we free all allocated structures.
  void clear() override;
  

 private:
  // This method allows us to build the "tree" from the "rechitsSet".
  void buildTree(const RecHitSet	&rechitsSet,
		   KDTreeLinkerAlgo	&tree);

 private:
  // Some const values. 
  const double	resPSpitch_;
  const double	resPSlength_;
  const double	ps1ToEcal_; // ratio : zEcal / zPS1
  const double	ps2ToEcal_; // ration : zEcal / zPS2

  // Data used by the KDTree algorithm : sets of PS and ECAL clusters.
  BlockEltSet		targetSet_;
  BlockEltSet		fieldClusterSet_;

  // Sets of rechits that compose the ECAL clusters. We differenctiate 
  // the rechits by their Z value.
  RecHitSet		rechitsNegSet_;
  RecHitSet		rechitsPosSet_;
  
  // Map of linked PS/ECAL clusters.
  BlockElt2BlockEltMap	target2ClusterLinks_;

  // Map of the ECAL clusters associated to a rechit.
  RecHit2BlockEltMap	rechit2ClusterLinks_;
    
  // KD trees
  KDTreeLinkerAlgo	treeNeg_;
  KDTreeLinkerAlgo	treePos_;
};

#endif /* !KDTreeLinkerPSEcal_h */
