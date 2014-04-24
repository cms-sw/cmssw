#ifndef RecoTracker_CkfPattern_BaseCkfTrajectoryBuilderFactor_h
#define RecoTracker_CkfPattern_BaseCkfTrajectoryBuilderFactor_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}

typedef edmplugin::PluginFactory< BaseCkfTrajectoryBuilder* (const edm::ParameterSet&, edm::ConsumesCollector& iC) > BaseCkfTrajectoryBuilderFactory;

#endif
