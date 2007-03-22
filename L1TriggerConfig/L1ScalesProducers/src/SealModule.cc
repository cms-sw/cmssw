#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTrivialProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTester.h"

DEFINE_FWK_EVENTSETUP_MODULE(L1ScalesTrivialProducer);
DEFINE_ANOTHER_FWK_MODULE(L1ScalesTester);
