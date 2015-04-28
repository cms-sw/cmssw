#include "DataFormats/HeavyIonEvent/interface/ClusterCompatibility.h"
using namespace reco;

ClusterCompatibility::ClusterCompatibility(float z0, int nHit, float chi):
  z0_(z0),
  nHit_(nHit),
  chi_(chi)
{}

ClusterCompatibility::ClusterCompatibility():
  z0_(0.),
  nHit_(0),
  chi_(0.)
{}

ClusterCompatibility::~ClusterCompatibility() {}
