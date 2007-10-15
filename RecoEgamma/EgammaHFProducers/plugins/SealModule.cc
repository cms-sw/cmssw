
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "HFEMClusterProducer.h"
#include "HFRecoEcalCandidateProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HFEMClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(HFRecoEcalCandidateProducer);
