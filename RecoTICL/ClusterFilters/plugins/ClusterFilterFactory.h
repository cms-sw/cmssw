#ifndef RecoHGCal_TICL_ClusterFilterFactory_h
#define RecoHGCal_TICL_ClusterFilterFactory_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "ClusterFilterBase.h"

typedef edmplugin::PluginFactory<ticl::ClusterFilterBase*(const edm::ParameterSet&)> ClusterFilterFactory;

#endif
