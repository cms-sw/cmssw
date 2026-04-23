#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/OfflineVertexHostCollection.h"
#include "DataFormats/VertexSoA/interface/TrackForVertexHostCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::ZVertexHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(OfflineVertexHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(TrackForVertexHostCollection);
