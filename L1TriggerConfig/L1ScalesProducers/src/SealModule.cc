#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTrivialProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuGMTScalesProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTester.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuScalesTester.h"

DEFINE_FWK_EVENTSETUP_MODULE(L1ScalesTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1MuTriggerScalesProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1MuTriggerPtScaleProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1MuGMTScalesProducer);

DEFINE_ANOTHER_FWK_MODULE(L1ScalesTester);
DEFINE_ANOTHER_FWK_MODULE(L1MuScalesTester);
