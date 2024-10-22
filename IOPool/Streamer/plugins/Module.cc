#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//New module to write events from Streamer files
#include "IOPool/Streamer/interface/StreamerOutputModule.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"

//new module to read events from Streamer files
#include "IOPool/Streamer/src/StreamerFileReader.h"

using EventStreamFileWriter = edm::streamer::StreamerOutputModule<edm::streamer::StreamerFileWriter>;
using NewEventStreamFileReader = edm::streamer::StreamerFileReader;

DEFINE_FWK_INPUT_SOURCE(NewEventStreamFileReader);

DEFINE_FWK_MODULE(EventStreamFileWriter);
