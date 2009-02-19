#ifndef CondTools_RunInfo_HLTScalerReaderFactory_h
#define CondTools_RunInfo_HLTScalerReaderFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondTools/RunInfo/interface/HLTScalerReaderBase.h"
#include <string>
namespace edm{
  class ParameterSet;
}
namespace lumi{
  typedef edmplugin::PluginFactory< lumi::HLTScalerReaderBase*(const edm::ParameterSet) > HLTScalerReaderFactory;
}
#endif
