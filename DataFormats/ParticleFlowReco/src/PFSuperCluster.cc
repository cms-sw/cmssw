#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Common/interface/PtrVector.h"


using namespace std;
using namespace reco;

PFSuperCluster::PFSuperCluster(const edm::PtrVector<reco::PFCluster>& clusters):
clusters_(clusters)
{  
}

void PFSuperCluster::reset() {
  
  PFCluster::reset();
  clusters_.clear();

}

PFSuperCluster& PFSuperCluster::operator=(const PFSuperCluster& other) {

  PFCluster::operator=((PFCluster)other); 
  clusters_ = other.clusters_;

  return *this;
}


std::ostream& reco::operator<<(std::ostream& out, 
                               const PFSuperCluster& cluster) {
  
  if(!out) return out;

  const math::XYZPoint&  pos = cluster.position();
  const PFCluster::REPPoint&  posrep = cluster.positionREP();
  const std::vector< reco::PFRecHitFraction >& fracs =
    cluster.recHitFractions();

  out<<"PFSuperCluster "
     <<", clusters: "<<cluster.clusters().size()
     <<", layer: "<<cluster.layer()
     <<"\tE = "<<cluster.energy()
     <<"\tXYZ: "
     <<pos.X()<<","<<pos.Y()<<","<<pos.Z()<<" | "
     <<"\tREP: "
     <<posrep.Rho()<<","<<posrep.Eta()<<","<<posrep.Phi()<<" | "
     <<fracs.size()<<" rechits";

  for(unsigned i=0; i<fracs.size(); i++) {
    // PFRecHit is not available, print the detID
    if( !fracs[i].recHitRef().isAvailable() )
      out<<cluster.printHitAndFraction(i)<<", ";
    else
      out<<fracs[i]<<", ";
  }

  
  return out;
}
