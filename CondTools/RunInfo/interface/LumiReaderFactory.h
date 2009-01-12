#ifndef CondTools_RunInfo_LumiReaderFactory_h
#define CondTools_RunInfo_LumiReaderFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondTools/RunInfo/interface/LumiReaderBase.h"
#include <string>
namespace edm{
  class ParameterSet;
}
namespace lumi{
  typedef edmplugin::PluginFactory< lumi::LumiReaderBase*(const edm::ParameterSet) > LumiReaderFactory;
}
#endif
