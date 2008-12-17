#include "FWCore/Framework/interface/MakerMacros.h"

// Pi0 source module
#include "HLTriggerOffline/special/src/DQMHLTSourcePi0.h"

DEFINE_ANOTHER_FWK_MODULE(DQMHLTSourcePi0);

// CSCTF Halo Muon Trigger Chain
#include "HLTriggerOffline/special/src/HaloTrigger.h"

DEFINE_ANOTHER_FWK_MODULE(HaloTrigger);
