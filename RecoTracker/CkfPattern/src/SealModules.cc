
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/KFTrackCandidateMaker.h"
using cms::KFTrackCandidateMaker;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(KFTrackCandidateMaker)
