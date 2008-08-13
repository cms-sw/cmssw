#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/src/TestConsumer.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//New module to write events from Streamer files
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"

//new module to read events from Streamer files
#include "IOPool/Streamer/interface/StreamerInputModule.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"

#include "IOPool/Streamer/interface/FRDEventFileWriter.h"
#include "IOPool/Streamer/interface/FRDEventOutputModule.h"

typedef edm::StreamerOutputModule<edm::StreamerFileWriter> EventStreamFileWriter;
typedef edm::StreamerInputModule<edm::StreamerFileReader> NewEventStreamFileReader;
typedef FRDEventOutputModule<FRDEventFileWriter> FRDStreamFileWriter;
typedef FRDEventOutputModule<FRDEventFileWriter> ErrorStreamFileWriter;

using edm::StreamerFileReader;
using edm::StreamerFileWriter;

DEFINE_FWK_INPUT_SOURCE(NewEventStreamFileReader);

DEFINE_FWK_MODULE(EventStreamFileWriter);

using namespace edm::serviceregistry;
using stor::HLTInfo;

DEFINE_FWK_SERVICE_MAKER(HLTInfo,ParameterSetMaker<HLTInfo>);

DEFINE_FWK_MODULE(FRDStreamFileWriter);
DEFINE_FWK_MODULE(ErrorStreamFileWriter);
