#include "RecoTracker/DebugTools/interface/CkfDebugTrackCandidateMaker.h"
#include "RecoTracker/DebugTools/interface/CkfDebugTrajectoryBuilderESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

using cms::CkfDebugTrackCandidateMaker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CkfDebugTrackCandidateMaker);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfDebugTrajectoryBuilderESProducer);
