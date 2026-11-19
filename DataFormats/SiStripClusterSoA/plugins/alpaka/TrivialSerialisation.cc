#include "DataFormats/SiStripClusterSoA/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClusterDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(sistrip::SiStripClusterHost, sistrip::SiStripClusterDevice);
