#ifndef __RecHitCleanerFactory_H__
#define __RecHitCleanerFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerBase.h"

typedef edmplugin::PluginFactory< RecHitCleanerBase* (const edm::ParameterSet&) > RecHitCleanerFactory;

#endif
