#ifndef RecoLuminosity_LumiProducer_DataPipeFactory_H
#define RecoLuminosity_LumiProducer_DataPipeFactory_H
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include <string>
namespace lumi{
  typedef edmplugin::PluginFactory< lumi:: DataPipe*( const std::string& ) > DataPipeFactory;
}
#endif
