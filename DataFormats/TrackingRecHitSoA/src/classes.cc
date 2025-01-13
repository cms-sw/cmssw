#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackingRecHitLayout<pixelTopology::Phase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackingRecHitLayout<pixelTopology::Phase2>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackingRecHitLayout<pixelTopology::HIonPhase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackingRecHitLayout<pixelTopology::Phase1Strip>>);
