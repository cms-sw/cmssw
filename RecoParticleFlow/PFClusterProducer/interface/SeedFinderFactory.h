#ifndef __SeedFinderFactory_H__
#define __SeedFinderFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

typedef edmplugin::PluginFactory< SeedFinderBase* (const edm::ParameterSet&) > SeedFinderFactory;

#endif
