#ifndef RecoHGCAL_TICL_LinkingAlgoFactory_h
#define RecoHGCAL_TICL_LinkingAlgoFactory_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "LinkingAlgoBase.h"

using LinkingAlgoFactory = edmplugin::PluginFactory<ticl::LinkingAlgoBase*(const edm::ParameterSet&)>;

#endif
