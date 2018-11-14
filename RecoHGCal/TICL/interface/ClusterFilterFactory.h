#ifndef RecoHGCal_TICL_ClusterFilterFactory_H
#define RecoHGCal_TICL_ClusterFilterFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoHGCal/TICL/interface/ClusterFilterBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory< ClusterFilterBase * (const edm::ParameterSet&) > ClusterFilterFactory;

#endif
