#include "DataFormats/METReco/interface/HaloClusterCandidateEE.h"

using namespace reco;
HaloClusterCandidateEE::HaloClusterCandidateEE()
{
  et =0;
  seed_et =0;
  seed_eta =0;
  seed_phi =0;
  seed_Z =0;
  seed_R =0;
  seed_time =0;
  timediscriminator =0;
  h2overe=0;
  nbearlycrystals=0;
  nblatecrystals=0;
  clustersize=0;
  ishalofrompattern=false;
  ishalofrompattern_hlt=false;
}


