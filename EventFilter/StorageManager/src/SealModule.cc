#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"

using edm::EventStreamHttpReader;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EventStreamHttpReader);

