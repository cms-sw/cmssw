#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::Phase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::Phase2>>);
// SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::HIonPhase1>>);