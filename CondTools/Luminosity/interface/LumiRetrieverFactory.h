#ifndef CondTools_Luminosity_LumiRetrieverFactory_h
#define CondTools_Luminosity_LumiRetrieverFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
#include <string>
namespace edm{
  class ParameterSet;
}
namespace lumi{
  typedef edmplugin::PluginFactory< lumi::LumiRetrieverBase*(const edm::ParameterSet) > LumiRetrieverFactory;
}
#endif
