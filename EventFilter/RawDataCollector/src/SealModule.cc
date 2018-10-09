#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "EventFilter/RawDataCollector/src/RawDataCollectorByLabel.h"
#include "EventFilter/RawDataCollector/interface/RawDataMapperByLabel.h"

using namespace edm::serviceregistry;

DEFINE_FWK_MODULE(RawDataCollectorByLabel);
DEFINE_FWK_MODULE(RawDataMapperByLabel);

