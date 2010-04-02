#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <sstream>
#include <iostream>
int main(int argc, char** argv){
  unsigned int runnumber;
  std::string lumifile;
  if(argc>2){
    std::istringstream iss(argv[1]);
    iss>>runnumber;
    lumifile=std::string(argv[2]);
  }else{
    std::cout<<"must specify a run and lumi file path"<<std::endl;
    return 0;
  }
  
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  //const std::string con("sqlite_file:pippo.db");
  const std::string con("oracle://cms_orcoff_prep/cms_lumi_dev_offline");
  const std::string authpath("/afs/cern.ch/user/x/xiezhen");
  //
  //fill runsummary data
  //
  std::cout<<"fill out runsummary data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("CMSRunSummary2DB",con));
  runptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
  runptr->setAuthPath(authpath);
  runptr->retrieveData(runnumber);
  //
  //fill hlt conf data
  //
  std::cout<<"fill out conf data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create("HLTConf2DB",con));
  //confptr->setSource("oracle://cms_omds_lb/CMS_HLT_V0");
  confptr->setSource("oracle://cms_omds_lb/CMS_HLT");
  confptr->setAuthPath(authpath);
  confptr->retrieveData(runnumber);

  //fill lhx data
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("Lumi2DB",con));
  ptr->setAuthPath(authpath);
  //ptr->setSource("rfio:/castor/cern.ch/cms/store/lumi/200912/CMS_LUMI_RAW_20091212_000124025_0001_1.root");
  ptr->setSource(lumifile);
  ptr->retrieveData(runnumber);
  //fill trg data
  std::cout<<"fill out trg data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRG2DB",con));
  trgptr->setAuthPath(authpath);
  trgptr->setSource("oracle://cms_omds_lb/CMS_GT_MON");
  trgptr->retrieveData(runnumber);
  //fill hlt scaler data
  
  std::cout<<"fill out hlt data"<<std::endl;
  std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLT2DB",con));
  hltptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
  hltptr->setAuthPath(authpath);
  hltptr->retrieveData(runnumber);
 
  return 0;
}
