#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryMaker.h"
#include "RecoTracker/CkfPattern/plugins/CkfTrajectoryBuilderESProducer.h"
#include "RecoTracker/CkfPattern/plugins/GroupedCkfTrajectoryBuilderESProducer.h"


#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"

using cms::CkfTrackCandidateMaker;
using cms::CkfTrajectoryMaker;


DEFINE_FWK_EVENTSETUP_MODULE(CkfTrajectoryBuilderESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(GroupedCkfTrajectoryBuilderESProducer);
DEFINE_FWK_MODULE(CkfTrackCandidateMaker);
DEFINE_FWK_MODULE(CkfTrajectoryMaker);
