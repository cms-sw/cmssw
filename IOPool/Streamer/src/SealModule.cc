#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/src/TestConsumer.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//New module to write events from Streamer files
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"

//new module to read events from Streamer files
#include "IOPool/Streamer/interface/StreamerInputModule.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"

typedef edm::StreamerOutputModule<edm::StreamerFileWriter> EventStreamFileWriter;
typedef edm::StreamerInputModule<edm::StreamerFileReader> NewEventStreamFileReader;

using edm::StreamerFileReader;
using edm::StreamerFileWriter;

//Old way of reading and writting Streamer files
typedef edm::EventStreamingModule<edmtest::TestConsumer> StreamTestConsumer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(StreamTestConsumer);

DEFINE_ANOTHER_FWK_INPUT_SOURCE(NewEventStreamFileReader);

DEFINE_ANOTHER_FWK_MODULE(EventStreamFileWriter);

using namespace edm::serviceregistry;
using stor::HLTInfo;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(HLTInfo,ParameterSetMaker<HLTInfo>);

