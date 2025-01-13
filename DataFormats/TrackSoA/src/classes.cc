#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"

using namespace reco;

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::Phase1>>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::Phase2>>);
// SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::HIonPhase1>>); //TODO: For the moment we live without HIons
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::Phase1Strip>>);
