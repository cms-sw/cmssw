#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"

using edm::EventStreamHttpReader;
using edm::DQMHttpSource;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EventStreamHttpReader);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(DQMHttpSource);

