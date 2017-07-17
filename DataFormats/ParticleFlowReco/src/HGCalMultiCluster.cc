#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

using namespace reco;

HGCalMultiCluster::HGCalMultiCluster(double energy,
                                     double x, double y, double z,
                                     ClusterCollection &thecls) :  
  PFCluster(PFLayer::HGCAL, energy, x, y, z),
  myclusters(thecls) {
  assert(myclusters.size() > 0 && "Invalid cluster collection, zero length.");
  }

  
