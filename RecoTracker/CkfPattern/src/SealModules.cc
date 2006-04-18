
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CombinatorialTrackFinder.h"
using cms::CombinatorialTrackFinder;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CombinatorialTrackFinder)
