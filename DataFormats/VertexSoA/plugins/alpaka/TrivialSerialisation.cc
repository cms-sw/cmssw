#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(reco::ZVertexSoACollection, "reco::ZVertexDeviceCollection");
