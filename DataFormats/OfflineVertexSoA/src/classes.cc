#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"

#include "DataFormats/VertexSoA/interface/VertexHostCollection.h"
#include "DataFormats/VertexSoA/interface/TrackHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(VertexHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(TrackHostCollection);
