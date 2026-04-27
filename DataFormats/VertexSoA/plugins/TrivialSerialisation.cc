#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/VertexSoA/interface/TrackForVertexHostCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::ZVertexHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(VertexHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(TrackForVertexHostCollection);
