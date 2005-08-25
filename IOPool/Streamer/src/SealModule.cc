#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputServiceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/src/TestProducer.h"
#include "IOPool/Streamer/interface/EventStreamInput.h"
#include "IOPool/Streamer/src/TestConsumer.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"

// using edm::EventStreamInput;
typedef edm::EventStreamingModule<edmtest::TestConsumer> StreamTestConsumer;
typedef edm::EventStreamInput<edmtestp::TestProducer> StreamTestProducer;

DEFINE_SEAL_MODULE();
// DEFINE_ANOTHER_FWK_INPUT_SERVICE(EventStreamInput)
DEFINE_ANOTHER_FWK_INPUT_SERVICE(StreamTestProducer)
DEFINE_ANOTHER_FWK_MODULE(StreamTestConsumer)

