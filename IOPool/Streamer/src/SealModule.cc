#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/src/TestConsumer.h"
#include "IOPool/Streamer/interface/FragmentInput.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/src/EventStreamFileReader.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
//#include "IOPool/Streamer/src/EventStreamFileWriter.h"


#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"

//typedef edm::EventStreamFileWriter StreamTestConsumer;

typedef edm::EventStreamingModule<edmtest::TestConsumer> StreamTestConsumer;

//using edm::EventStreamFileWriter;

using edmtestp::StreamerFileReader;
using edm::StreamerFileWriter;
using stor::FragmentInput;
using edmtestp::EventStreamFileReader;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(StreamTestConsumer);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FragmentInput);

DEFINE_ANOTHER_FWK_INPUT_SOURCE(EventStreamFileReader);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(StreamerFileReader);

//DEFINE_ANOTHER_FWK_MODULE(EventStreamFileWriter);
DEFINE_ANOTHER_FWK_MODULE(StreamerFileWriter);

using namespace edm::serviceregistry;
using stor::HLTInfo;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(HLTInfo,ParameterSetMaker<HLTInfo>)
