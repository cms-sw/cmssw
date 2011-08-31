#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//New module to write events from Streamer files
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"

//new module to read events from Streamer files
#include "IOPool/Streamer/src/StreamerFileReader.h"

#include "IOPool/Streamer/interface/FRDEventFileWriter.h"
#include "IOPool/Streamer/interface/FRDEventOutputModule.h"

typedef edm::StreamerOutputModule<edm::StreamerFileWriter> EventStreamFileWriter;
typedef edm::StreamerFileReader NewEventStreamFileReader;
typedef FRDEventOutputModule<FRDEventFileWriter> FRDStreamFileWriter;
typedef FRDEventOutputModule<FRDEventFileWriter> ErrorStreamFileWriter;

using edm::StreamerFileReader;
using edm::StreamerFileWriter;

DEFINE_FWK_INPUT_SOURCE(NewEventStreamFileReader);

DEFINE_FWK_MODULE(EventStreamFileWriter);

DEFINE_FWK_MODULE(FRDStreamFileWriter);
DEFINE_FWK_MODULE(ErrorStreamFileWriter);
