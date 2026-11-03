#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/VertexGNNDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::ZVertexHost, reco::ZVertexSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(vertexgnn::TrackFeaturesHostCollection,
                                          vertexgnn::TrackFeaturesDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(vertexgnn::GNNOutputHostCollection, vertexgnn::GNNOutputDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(vertexgnn::GNNSlotHostCollection, vertexgnn::GNNSlotDeviceCollection);
