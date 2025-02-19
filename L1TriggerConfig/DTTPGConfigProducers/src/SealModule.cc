#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTrivialProducer.h"
#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigDBProducer.h"
#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTester.h"


DEFINE_FWK_EVENTSETUP_MODULE(DTConfigTrivialProducer);
DEFINE_FWK_EVENTSETUP_MODULE(DTConfigDBProducer);
DEFINE_FWK_MODULE(DTConfigTester);
