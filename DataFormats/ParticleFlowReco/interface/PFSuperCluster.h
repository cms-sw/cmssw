#ifndef DataFormats_ParticleFlowReco_PFSuperCluster_h
#define DataFormats_ParticleFlowReco_PFSuperCluster_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include <iostream>
#include <vector>



class PFSuperClusterAlgo;

namespace reco {

  /**\class PFSuperCluster
     \brief Particle flow cluster, see clustering algorithm in PFSuperClusterAlgo
     
     A particle flow supercluster is constructed from clusters.
     This calculation is performed in PFSuperClusterAlgo.

     \author Chris Tully
     \date   July 2012
  */
  class PFSuperCluster : public PFCluster {
  public:


    PFSuperCluster(){}

    /// constructor
    PFSuperCluster(const edm::PtrVector<reco::PFCluster>& clusters);
   
    /// resets clusters parameters
    void reset();
    
    /// vector of clusters
    const edm::PtrVector< reco::PFCluster >& clusters() const 
      { return clusters_; }
    
    PFSuperCluster& operator=(const PFSuperCluster&);
    
    friend    std::ostream& operator<<(std::ostream& out, 
				       const PFSuperCluster& cluster);

  private:
    
    /// vector of clusters
    edm::PtrVector< reco::PFCluster >  clusters_;
    
    friend class ::PFSuperClusterAlgo;
  };
}

#endif
