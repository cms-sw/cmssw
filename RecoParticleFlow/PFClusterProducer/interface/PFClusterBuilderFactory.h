#ifndef __PFClusterBuilderFactory_H__
#define __PFClusterBuilderFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"

typedef edmplugin::PluginFactory< PFClusterBuilderBase* (const edm::ParameterSet&) > PFClusterBuilderFactory;

#endif
