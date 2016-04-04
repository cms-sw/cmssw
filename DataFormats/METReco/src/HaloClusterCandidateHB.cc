#include "DataFormats/METReco/interface/HaloClusterCandidateHB.h"

using namespace reco;
HaloClusterCandidateHB::HaloClusterCandidateHB()
{
  et =0;
  seed_et =0;
  seed_eta =0;
  seed_phi =0;
  seed_Z =0;
  seed_R =0;
  seed_time =0;
  timediscriminatoritbh =0;
  timediscriminatorotbh =0;
  eoverh=0;
  nbtowersineta=0;
  etstrip_phiseedplus1=0;
  etstrip_phiseedminus1=0;
  ishalofrompattern = false;
  ishalofrompattern_hlt = false;
}
