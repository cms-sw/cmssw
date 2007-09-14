#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTrivialProducer.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTester.h"

DEFINE_FWK_EVENTSETUP_MODULE(L1GtFactorsTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtParametersTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtBoardMapsTrivialProducer);

DEFINE_ANOTHER_FWK_MODULE(L1GtFactorsTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtParametersTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtBoardMapsTester);
