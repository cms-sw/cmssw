#ifndef RecoParticleFlow_PFClusterProducer_PFHcalSuperClusterInit_h
#define RecoParticleFlow_PFClusterProducer_PFHcalSuperClusterInit_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"

#include <iostream>
#include <vector>



namespace reco{

  class PFHcalSuperClusterInit {
  public:

    PFHcalSuperClusterInit(){}
    ~PFHcalSuperClusterInit(){}

    void initialize( reco::PFSuperCluster &supercluster, edm::PtrVector<reco::PFCluster> const & clusters   );


  };
}
#endif



