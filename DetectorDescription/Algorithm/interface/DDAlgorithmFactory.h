#ifndef DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
# define DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H

#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<DDAlgorithm *(void)> DDAlgorithmFactory;

#endif // DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
