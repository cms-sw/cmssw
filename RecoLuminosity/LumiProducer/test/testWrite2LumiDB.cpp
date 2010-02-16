#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  const std::string con("sqlite_file:pippo.db");
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("LumiDummy2DB",con));
  ptr->retrieveRun(1234);
  return 0;
}
