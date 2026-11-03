#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::ZVertexHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(vertexgnn::TrackFeaturesHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(vertexgnn::GNNOutputHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(vertexgnn::GNNSlotHostCollection);
