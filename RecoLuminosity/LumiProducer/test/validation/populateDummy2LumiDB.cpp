#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
//#include <boost/program_options.hpp>
//ToDo:change to use command options
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  //const std::string con("sqlite_file:pippo.db");
  const std::string con("oracle://devdb10/cms_xiezhen_dev");
  const std::string authpath("/afs/cern.ch/user/x/xiezhen");
  std::cout<<"fill out hlx data"<<std::endl;
  unsigned int fakerunnumber=1234;
  //fill cmsrunsummary data
  std::cout<<"fill out runsummary data"<<std::endl;
  std::unique_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("CMSRunSummaryDummy2DB",con));
  runptr->setAuthPath(authpath);
  runptr->retrieveData(fakerunnumber);
  
  //fill lhx data
  std::unique_ptr<lumi::DataPipe> lumiptr(lumi::DataPipeFactory::get()->create("LumiDummy2DB",con));
  lumiptr->setAuthPath(authpath);
  lumiptr->retrieveData(fakerunnumber);
   
  //fill trg data
  std::cout<<"fill out trg data"<<std::endl;
  std::unique_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRGDummy2DB",con));
  trgptr->setAuthPath(authpath);
  trgptr->retrieveData(fakerunnumber);
   
  //fill hlt conf data
  std::cout<<"fill out conf data"<<std::endl;
  std::unique_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create("HLTConfDummy2DB",con));
  confptr->setAuthPath(authpath);
  confptr->retrieveData(fakerunnumber);
  
  //fill hlt scaler data
  std::cout<<"fill out hlt data"<<std::endl;
  std::unique_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLTDummy2DB",con));
  hltptr->setAuthPath(authpath);
  hltptr->retrieveData(fakerunnumber);
  return 0;
}
