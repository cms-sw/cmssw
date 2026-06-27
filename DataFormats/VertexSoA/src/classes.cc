#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/VertexSoA/interface/TrackForVertexHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::ZVertexHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(VertexHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(TrackForVertexHostCollection);
