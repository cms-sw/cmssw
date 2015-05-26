#include "DataFormats/HeavyIonEvent/interface/ClusterCompatibility.h"
using namespace reco;

ClusterCompatibility::ClusterCompatibility():
  nValidPixelHits_(0),
  z0_(),
  nHit_(),
  chi_()
{}

ClusterCompatibility::~ClusterCompatibility() {}

void
ClusterCompatibility::append(float z0, int nHit, float chi) {
  z0_.push_back(z0);
  nHit_.push_back(nHit);
  chi_.push_back(chi);
}
