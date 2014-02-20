#ifndef __InitialClusteringStepFactory_H__
#define __InitialClusteringStepFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

typedef edmplugin::PluginFactory< InitialClusteringStepBase* (const edm::ParameterSet&) > InitialClusteringStepFactory;

#endif
