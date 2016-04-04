#include "DataFormats/METReco/interface/HaloClusterCandidateHE.h"

using namespace reco;
HaloClusterCandidateHE::HaloClusterCandidateHE()
{
  et =0;
  seed_et =0;
  seed_eta =0;
  seed_phi =0;
  seed_Z =0;
  seed_R =0;
  seed_time =0;
  timediscriminator =0;
  eoverh=0;
  h1overh123=0;
  etstrip_phiseedplus1=0;
  etstrip_phiseedminus1=0;
  clustersize=0;
  ishalofrompattern = false;
  ishalofrompattern_hlt = false;
}
