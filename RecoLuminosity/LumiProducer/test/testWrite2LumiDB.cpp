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
  //std::cout<<"fill out hlx data"<<std::endl;
  //
  //fill runsummary data
  //
  std::cout<<"fill out runsummary data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("CMSRunSummary2DB",con));
  runptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
  runptr->setAuthPath(authpath);
  runptr->retrieveData(129265);
  
  //
  //fill hlt conf data
  //
  std::cout<<"fill out conf data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create("HLTConf2DB",con));
  confptr->setSource("oracle://cms_omds_lb/CMS_HLT");
  confptr->setAuthPath(authpath);
  confptr->retrieveData(129265);
  
  //fill lhx data
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("Lumi2DB",con));
  ptr->setAuthPath(authpath);
  ptr->setSource("rfio:/castor/cern.ch/cms/store/lumi/200912/CMS_LUMI_RAW_20091212_000124025_0001_1.root");
  ptr->retrieveData(124025);
  
  //fill trg data
  std::cout<<"fill out trg data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRG2DB",con));
  trgptr->setAuthPath(authpath);
  trgptr->setSource("oracle://cms_omds_lb/CMS_GT_MON");
  trgptr->retrieveData(129265);
    
  //fill hlt scaler data
  std::cout<<"fill out hlt data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLT2DB",con));
  hltptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
  hltptr->setAuthPath(authpath);
  hltptr->retrieveData(129265);

  return 0;
}
