#ifndef RecoHGCAL_TICL_SeedingRegionAlgoFactory_h
#define RecoHGCAL_TICL_SeedingRegionAlgoFactory_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SeedingRegionAlgoBase.h"

using SeedingRegionAlgoFactory =
    edmplugin::PluginFactory<ticl::SeedingRegionAlgoBase*(const edm::ParameterSet&, edm::ConsumesCollector&)>;

#endif
