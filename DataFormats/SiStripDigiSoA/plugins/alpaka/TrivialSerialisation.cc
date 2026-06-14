#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiHost.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(sistrip::SiStripDigiHost, sistrip::SiStripDigiDevice);
