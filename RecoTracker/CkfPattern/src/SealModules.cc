#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
using cms::CkfTrackCandidateMaker;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CkfTrackCandidateMaker)
