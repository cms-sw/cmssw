#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  const std::string con("sqlite_file:pippo.db");
  //fill lhx data
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("LumiDummy2DB",con));
  ptr->retrieveRun(1234);
  //fill hlt data
   std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLTDummy2DB",con));
  hltptr->retrieveRun(1234);
  //fill trg data
  std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRGDummy2DB",con));
  trgptr->retrieveRun(1234);
  //fill runsummary data
  std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("RunSummaryDummy2DB",con));
  runptr->retrieveRun(1234);
  return 0;
}
