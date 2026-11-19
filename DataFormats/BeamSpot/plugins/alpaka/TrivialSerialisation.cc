#include "DataFormats/BeamSpot/interface/BeamSpotHost.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(BeamSpotHost, BeamSpotDevice);
