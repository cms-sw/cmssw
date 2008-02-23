#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtStableParametersTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTrivialProducer.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlProducer.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtStableParametersTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTester.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriter.h"

DEFINE_FWK_EVENTSETUP_MODULE(L1GtStableParametersTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtParametersTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtFactorsTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtBoardMapsTrivialProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1GtTriggerMenuXmlProducer);

DEFINE_ANOTHER_FWK_MODULE(L1GtStableParametersTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtParametersTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtFactorsTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtBoardMapsTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtTriggerMenuTester);
DEFINE_ANOTHER_FWK_MODULE(L1GtVhdlWriter);
DEFINE_ANOTHER_FWK_MODULE(L1GtVmeWriter);

