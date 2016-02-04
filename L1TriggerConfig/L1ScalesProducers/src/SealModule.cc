#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTrivialProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScalesProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesOnlineProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleOnlineProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuGMTScalesProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTester.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuScalesTester.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScaleTester.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScalesGenerator.h"

DEFINE_FWK_EVENTSETUP_MODULE(L1ScalesTrivialProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloInputScalesProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1MuTriggerScalesProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1MuTriggerPtScaleProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1MuTriggerScalesOnlineProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1MuTriggerPtScaleOnlineProducer);
DEFINE_FWK_EVENTSETUP_MODULE(L1MuGMTScalesProducer);

DEFINE_FWK_MODULE(L1ScalesTester);
DEFINE_FWK_MODULE(L1MuScalesTester);
DEFINE_FWK_MODULE(L1CaloInputScaleTester);
DEFINE_FWK_MODULE(L1CaloInputScalesGenerator);
