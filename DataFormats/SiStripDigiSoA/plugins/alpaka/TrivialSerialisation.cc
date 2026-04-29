#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(sistrip::SiStripDigiHost, sistrip::SiStripDigiDevice);
