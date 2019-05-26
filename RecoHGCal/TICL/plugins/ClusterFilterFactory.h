#ifndef RecoHGCal_TICL_ClusterFilterFactory_H
#define RecoHGCal_TICL_ClusterFilterFactory_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "ClusterFilterBase.h"

typedef edmplugin::PluginFactory<ticl::ClusterFilterBase*(const edm::ParameterSet&)> ClusterFilterFactory;

#endif
