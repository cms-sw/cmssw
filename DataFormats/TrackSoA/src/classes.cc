#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"

using namespace reco;
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(TrackPortableCollectionHostPhase1);
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(TrackPortableCollectionHostPhase2);
// SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<TrackLayout<pixelTopology::HIonPhase1>>); //TODO: For the moment we live without HIons
