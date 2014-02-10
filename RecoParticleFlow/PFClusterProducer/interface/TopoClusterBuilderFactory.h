#ifndef __TopoClusterBuilderFactory_H__
#define __TopoClusterBuilderFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderBase.h"

typedef edmplugin::PluginFactory< TopoClusterBuilderBase* (const edm::ParameterSet&) > TopoClusterBuilderFactory;

#endif
