#ifndef __PFCPositionCalculatorFactory_H__
#define __PFCPositionCalculatorFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

typedef edmplugin::PluginFactory< PFCPositionCalculatorBase* (const edm::ParameterSet&) > PFCPositionCalculatorFactory;

#endif
