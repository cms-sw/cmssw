#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"

using edm::EventStreamHttpReader;
using edm::DQMHttpSource;

DEFINE_FWK_INPUT_SOURCE(EventStreamHttpReader);
DEFINE_FWK_INPUT_SOURCE(DQMHttpSource);

