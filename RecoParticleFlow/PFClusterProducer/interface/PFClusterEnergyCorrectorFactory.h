#ifndef __PFClusterEnergyCorrectorFactory_H__
#define __PFClusterEnergyCorrectorFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

typedef edmplugin::PluginFactory< PFClusterEnergyCorrectorBase* (const edm::ParameterSet&) > PFClusterEnergyCorrectorFactory;

#endif
