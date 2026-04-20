#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"

#include "DataFormats/OfflineVertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/TrackHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(VertexHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(TrackHostCollection);
