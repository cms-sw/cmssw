#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilderESProducer.h"
#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilderESProducer.h"


#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

using cms::CkfTrackCandidateMaker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(GroupedCkfTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_MODULE(CkfTrackCandidateMaker);
