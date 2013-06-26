#ifndef Alignment_ReferenceTrajectories_TrajectoryFactoryPlugin_h
#define Alignment_ReferenceTrajectories_TrajectoryFactoryPlugin_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include <string>

/// A PluginFactory that produces factories that inherit from TrajectoryFactoryBase.


typedef edmplugin::PluginFactory< TrajectoryFactoryBase *( const edm::ParameterSet & ) >
                   TrajectoryFactoryPlugin;

#endif
