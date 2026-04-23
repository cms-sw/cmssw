#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/OfflineVertexDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/TrackForVertexDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::ZVertexSoACollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(OfflineVertexDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(TrackForVertexDeviceCollection);
