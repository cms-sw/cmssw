#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FastSimulation/Tracking/interface/GSTrackCandidateMaker.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GSTrackCandidateMaker);

#include "FastSimulation/Tracking/interface/PixelTracksMaker.h"
DEFINE_ANOTHER_FWK_MODULE(PixelTracksMaker);
