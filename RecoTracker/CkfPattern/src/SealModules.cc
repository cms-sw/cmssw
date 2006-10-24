#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilderESProducer.h"
#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilderESProducer.h"

//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

using cms::CkfTrackCandidateMaker;

EVENTSETUP_DATA_REG(TrackerTrajectoryBuilder);
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(GroupedCkfTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_MODULE(CkfTrackCandidateMaker);
