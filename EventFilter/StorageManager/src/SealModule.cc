#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "EventFilter/StorageManager/src/StreamerI2OWriter.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"

typedef edm::StreamerOutputModule<edm::StreamerI2OWriter> I2OStreamConsumer;

using edm::StreamerI2OWriter;
using edm::EventStreamHttpReader;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(I2OStreamConsumer);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EventStreamHttpReader);

