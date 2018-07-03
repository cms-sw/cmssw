#include "DataFormats/METReco/interface/HaloClusterCandidateECAL.h"

using namespace reco;
HaloClusterCandidateECAL::HaloClusterCandidateECAL() :  
  et(0),
  seed_et(0),
  seed_eta(0),
  seed_phi(0),
  seed_Z(0),
  seed_R(0),
  seed_time(0),
  timediscriminator(0),
  ishalofrompattern(false),
  ishalofrompattern_hlt(false),
  hovere(0),
  numberofcrystalsineta(0),
  etstrip_iphiseedplus1(0),
  etstrip_iphiseedminus1(0),
  h2overe(0),
  nbearlycrystals(0),
  nblatecrystals(0),
  clustersize(0)
{
}
