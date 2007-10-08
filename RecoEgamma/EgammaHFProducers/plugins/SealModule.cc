
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "HFClusterProducer.h"
#include "HFRecoEcalCandidateProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HFClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(HFRecoEcalCandidateProducer);
