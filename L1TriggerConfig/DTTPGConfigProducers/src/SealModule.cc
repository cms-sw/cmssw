#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTester.h"
#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTrivialProducer.h"

DEFINE_FWK_EVENTSETUP_MODULE(DTConfigTrivialProducer);
DEFINE_FWK_MODULE(DTConfigTester);
