#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "EventFilter/StorageManager/src/I2OConsumer.h"

typedef edm::EventStreamingModule<edmtest::I2OConsumer> I2OTestConsumer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(I2OTestConsumer)

