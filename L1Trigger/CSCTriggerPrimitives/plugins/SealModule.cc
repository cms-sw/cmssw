#include "FWCore/Framework/interface/MakerMacros.h"

#include <L1Trigger/CSCTriggerPrimitives/plugins/CSCTriggerPrimitivesProducer.h>
#include "L1Trigger/CSCTriggerPrimitives/src/CSCDigiSuppressor.h"

DEFINE_FWK_MODULE(CSCTriggerPrimitivesProducer);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiSuppressor);
