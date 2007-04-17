#include "FWCore/Framework/interface/MakerMacros.h"
 
#include "RecoTracker/DebugTools/interface/TestHits.h"
#include "RecoTracker/DebugTools/interface/TestTrackHits.h"
#include "RecoTracker/DebugTools/interface/CkfDebugTrajectoryBuilderESProducer.h"
#include "RecoTracker/DebugTools/interface/CkfDebugTrackCandidateMaker.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

using cms::CkfDebugTrackCandidateMaker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestHits);
DEFINE_ANOTHER_FWK_MODULE(TestTrackHits);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfDebugTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_MODULE(CkfDebugTrackCandidateMaker);
