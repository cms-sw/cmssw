#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClusterDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(sistrip::SiStripClusterHost, sistrip::SiStripClusterDevice);
