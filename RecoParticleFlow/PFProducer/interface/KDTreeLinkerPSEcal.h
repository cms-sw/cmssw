#ifndef KDTreeLinkerPSEcal_h
#define KDTreeLinkerPSEcal_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"

#include <vector>


namespace KDTreeLinker
{
  // Class that allows to use the KDTreeLinker optimization for PS
  // and Ecal clusters.
  class KDTreeLinkerPSEcal
  {
  public:
    // For the meaning of phiOffset and ecalDiameter fields, see bellow.
    KDTreeLinkerPSEcal(double phiOffset = 0.25,
		       double ecalDiameter = 0.04,
		       bool debug = false);
    ~KDTreeLinkerPSEcal();
  
    void setPhiOffset(double phiOffset);
    void setEcalDiameter(double ecalDiameter);
    double getPhiOffset() const;
    double getEcalDiameter() const;
  
    void setDebug(bool isDebug);
  
    // This 2 methods are used to build the lists of data on which the 
    // KDTreeLinkerAlgo will work.
    void insertTrack(reco::PFBlockElement* track);
    void insertCluster(const reco::PFBlockElement* cluster,
		       const std::vector<reco::PFRecHitFraction> &fraction);
  
    // Build the KDTree from ths lists of tracks and Ecal clusters.
    void buildTree();
  
    // Search for all linked tracks/clusters and save the links in a map.
    // After that, we may call isLinked() method.
    void searchLinks();
  
    // Save all founded links in the PFMultitracks field in each track.
    void updateTracksWithLinks();
  
    // This method clears ALL allocated structures (KDTree, maps, sets...).
    // After the call to clear(), isCorrectTrack(), isEcalCluster() and some
    // other methods will not work well.
    void clear();
  
  
/*     // Here, we may check if the Track/Cluster has been selected for the KDTree */
/*     // processing, i.e. the previous insert methods has been called on thus elements. */
/*     bool isCorrectTrack(reco::PFBlockElement* track) const; */
/*     bool isEcalCluster(const reco::PFBlockElement* cluster) const; */
  
/*     // Check if the track and the cluster are linked. Should not be called after clear(). */
/*     bool isLinked(reco::PFBlockElement* track, */
/* 		  const reco::PFBlockElement* cluster) const; */
  
/*     // Print all linked clusters to the specified track. */
/*     void printTrackLinks(reco::PFBlockElement* track); */
  
  private:
    // Usually, phi is between -Pi and +Pi. But phi space is circular, that's why an element 
    // with phi = 3.13 and another with phi = -3.14 are close. To solve this problem, during  
    // the kdtree building step, we duplicate some elements close enough to +Pi (resp -Pi) by
    // substracting (adding) 2Pi. This field define the threshold of this operation.
    double		phiOffset_;
  
    // When we search for the closest rechits, we approximate a maximal size envelope of rechits
    // to find all candidates. For this purpose, we need a maximal size of an ECAL cristal.
    double		ecalDiameter_;
  
    bool		debug_;
  
    // Data used by the KDTree algorithm : sets of tracks and clusters.
    BlockEltSet		tracksSet_;
    BlockEltSet_const	clustersSet_;
  
    // Set of all rechits that compose the clusters.
    RecHitSet		rechitsSet_;
  
    // Map of clusters associated to a rechit.
    RecHitClusterMap	rhClustersLinks_;
  
    // Map of linked tracks/clusters.
    BlockEltClusterMap	trackClusterLinks_;
  
    KDTreeLinkerAlgo	tree_;
  };
}

#endif /* !KDTreeLinkerPSEcal_h */
