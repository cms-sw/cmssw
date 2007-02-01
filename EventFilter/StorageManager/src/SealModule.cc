#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
//#include "EventFilter/StorageManager/src/I2OConsumer.h"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "EventFilter/StorageManager/src/StreamerI2OWriter.h"

//typedef edm::EventStreamingModule<edmtest::I2OConsumer> I2OTestConsumer;
typedef edm::StreamerOutputModule<edm::StreamerI2OWriter> I2OStreamConsumer;
using edmtestp::EventStreamHttpReader;

using edm::StreamerI2OWriter;

DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(I2OTestConsumer);
DEFINE_ANOTHER_FWK_MODULE(I2OStreamConsumer);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EventStreamHttpReader);

