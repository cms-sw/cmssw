#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include <sstream>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <cstdio>

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
  
  clock_t startClock, endClock;
  double elapsedTime;
  time_t t1,t2;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  //const std::string con("sqlite_file:wbm.db");
  //const std::string con("oracle://devdb10/cms_xiezhen_dev");
  const std::string con("oracle://cms_orcoff_prep/cms_lumi_dev_offline");
  const std::string authpath("/afs/cern.ch/user/x/xiezhen");
  //fill lhx data
  
  std::cout<<"filling hlx/dip data"<<std::endl;
  try{
    std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("Lumi2DB",con));
    ptr->setAuthPath(authpath);
    ptr->setSource(lumifile);
    startClock=clock();
    time(&t1);
    ptr->retrieveData(runnumber);
    time(&t2);
    endClock=clock();
  }catch(...){
    std::cout<<"problem in loading run "<<runnumber<<" skip "<<std::endl;
    throw;
  }
  printf("Elaspsed time %fs\n",difftime(t2,t1));
  elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
  std::cout<<"CPU Time taken in seconds : "<<elapsedTime<<std::endl;
  
  //
  //fill runsummary data
  //
  
  try{
    std::cout<<"fill out runsummary data"<<std::endl;
    std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("CMSRunSummary2DB",con));
    //runptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
    runptr->setSource("oracle://cms_orcoff_prod/CMS_RUNINFO");
    runptr->setAuthPath(authpath);
    startClock=clock();
    time(&t1);
    runptr->retrieveData(runnumber);
    time(&t2);
    endClock=clock();
  }catch(...){
    std::cout<<"problem in loading run "<<runnumber<<" skip "<<std::endl;
    throw;
  }
  printf("Elaspsed time %fs\n",difftime(t2,t1));
  elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
  std::cout<<"CPU Time taken in seconds : "<<elapsedTime<<std::endl;
  
  //
  //fill hlt conf data
  //
  try{
    std::cout<<"fill out conf data"<<std::endl;
    std::auto_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create("HLTConf2DB",con));
    //confptr->setSource("oracle://cms_omds_lb/CMS_HLT");
    confptr->setSource("oracle://cms_orcoff_prod/CMS_HLT");
    confptr->setAuthPath(authpath);
    startClock=clock();
    time(&t1);
    confptr->retrieveData(runnumber);
    time(&t2);
    endClock=clock();
  }catch(...){
    std::cout<<"problem in loading run "<<runnumber<<" skip "<<std::endl;
    throw;
  }
  printf("Elaspsed time %fs\n",difftime(t2,t1));
  elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
  std::cout<<"CPU Time taken in seconds : "<<elapsedTime<<std::endl;
  //fill trg data
  try{
    std::cout<<"fill out trg data from WBM"<<std::endl;
    std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRGWBM2DB",con));
    trgptr->setAuthPath(authpath);
    //trgptr->setSource("oracle://cms_omds_lb/CMS_GT_MON");
    //trgptr->setSource("oracle://cms_orcoff_prod/CMS_GT_MON");
    trgptr->setSource("oracle://cms_orcoff_prod/CMS_GT_MON");
    startClock=clock();
    time(&t1);
    trgptr->retrieveData(runnumber);
    time(&t2);
    endClock=clock();
    
  }catch(...){
    std::cout<<"problem in loading run "<<runnumber<<" skip "<<std::endl;
    throw;
  }
  printf("Elaspsed time %fs\n",difftime(t2,t1));
  elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
  std::cout<<"CPU Time taken in seconds : "<<elapsedTime<<std::endl;
  //fill hlt scaler data
  
  try{
    std::cout<<"fill out hlt data"<<std::endl;
    std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLT2DB",con));
    //hltptr->setSource("oracle://cms_omds_lb/CMS_RUNINFO");
    hltptr->setSource("oracle://cms_orcoff_prod/CMS_RUNINFO");
    hltptr->setAuthPath(authpath);
    startClock=clock();
    time(&t1);
    hltptr->retrieveData(runnumber);
    time(&t2);
    endClock=clock();
  }catch(...){
    std::cout<<"problem in loading run "<<runnumber<<" skip "<<std::endl;
    throw;
  }
  printf("Elaspsed time %fs\n",difftime(t2,t1));
  std::cout<<"CPU Time taken in seconds : "<<elapsedTime<<std::endl;
  
  return 0;
}
