#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  const std::string con("sqlite_file:pippo.db");
  std::unique_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("HLTConfDummy2DB",con));
  unsigned int hltconfId=5678;
  ptr->retrieveData(hltconfId);
  return 0;
}
