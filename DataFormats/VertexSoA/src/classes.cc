#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::ZVertexHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(vertexgnn::TrackFeaturesHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(vertexgnn::GNNOutputHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(vertexgnn::GNNSlotHostCollection);
