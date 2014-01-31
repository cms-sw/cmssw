#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryMaker.h"


#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

using cms::CkfTrackCandidateMaker;
using cms::CkfTrajectoryMaker;

DEFINE_FWK_MODULE(CkfTrackCandidateMaker);
DEFINE_FWK_MODULE(CkfTrajectoryMaker);

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilder.h"

DEFINE_EDM_PLUGIN(BaseCkfTrajectoryBuilderFactory, CkfTrajectoryBuilder, "CkfTrajectoryBuilder");
DEFINE_EDM_PLUGIN(BaseCkfTrajectoryBuilderFactory, GroupedCkfTrajectoryBuilder, "GroupedCkfTrajectoryBuilder");
