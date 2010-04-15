#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "EventFilter/StorageManager/src/OnlineHttpReader.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"

using edm::EventStreamHttpReader;
using edm::OnlineHttpReader;
using edm::DQMHttpSource;

DEFINE_FWK_INPUT_SOURCE(EventStreamHttpReader);
DEFINE_FWK_INPUT_SOURCE(OnlineHttpReader);
DEFINE_FWK_INPUT_SOURCE(DQMHttpSource);

