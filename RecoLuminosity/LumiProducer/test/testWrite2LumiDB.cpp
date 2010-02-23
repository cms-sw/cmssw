#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  //const std::string con("sqlite_file:pippo.db");
  const std::string con("oracle://devdb10/cms_xiezhen_dev");
  const std::string authpath("/afs/cern.ch/user/x/xiezhen");
  std::cout<<"fill out hlx data"<<std::endl;
  //fill lhx data
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("LumiDummy2DB",con));
  ptr->setAuthPath(authpath);
  ptr->setSource("rfio:/castor/cern.ch/cms/store/lumi/200912/CMS_LUMI_RAW_20091212_000124025_0001_1.root");
  ptr->retrieveData(124025);

  //fill trg data
  std::cout<<"fill out trg data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRGDummy2DB",con));
  trgptr->setAuthPath(authpath);
  trgptr->retrieveData(124025);

  /*
  //fill hlt data
  std::cout<<"fill out hlt data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLTDummy2DB",con));
  hltptr->setAuthPath(authpath);
  hltptr->retrieveData(1234);
  
  //fill runsummary data
  std::cout<<"fill out runsummary data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("RunSummaryDummy2DB",con));
  runptr->setAuthPath(authpath);
  runptr->retrieveData(1234);
  */
  return 0;
}
