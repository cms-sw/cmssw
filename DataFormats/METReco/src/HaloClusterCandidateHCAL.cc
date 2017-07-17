#include "DataFormats/METReco/interface/HaloClusterCandidateHCAL.h"

using namespace reco;
HaloClusterCandidateHCAL::HaloClusterCandidateHCAL() :
  et(0),
  seed_et(0),
  seed_eta(0),
  seed_phi(0),
  seed_Z(0),
  seed_R(0),
  seed_time(0),
  ishalofrompattern(false),
  ishalofrompattern_hlt(false),
  eoverh(0),
  etstrip_phiseedplus1(0),
  etstrip_phiseedminus1(0),
  nbtowersineta(0),
  timediscriminatoritbh(0),
  timediscriminatorotbh(0),
  h1overh123(0),
  clustersize(0),
  timediscriminator(0)
{
}
