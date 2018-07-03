#ifndef DD_CORE_PLUGIN_DD_ALGORITHM_FACTORY_H
# define DD_CORE_PLUGIN_DD_ALGORITHM_FACTORY_H

#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

class DDAlgorithm;

typedef edmplugin::PluginFactory<DDAlgorithm *(void)> DDAlgorithmFactory;

#endif // DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
