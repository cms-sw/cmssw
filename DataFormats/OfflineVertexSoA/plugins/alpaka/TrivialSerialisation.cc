#include "DataFormats/OfflineVertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/alpaka/TrackDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(VertexDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(TrackDeviceCollection);
