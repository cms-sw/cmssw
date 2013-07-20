#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
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
  
  //std::string lumidb("oracle://CMSDEVR_LB/CMS_LUMI");
  std::string lumidb("oracle://cms_orcon_prod/cms_lumi_prod");
  std::string runinfodb("oracle://cms_omds_lb/CMS_RUNINFO");
  std::string trgdb("oracle://cms_omds_lb/CMS_GT_MON");
  std::string hltconfdb("oracle://cms_omds_lb/CMS_HLT");
  const std::string authpath("/home/lumidb/auth/writer");
  
  clock_t startClock, endClock;
  double elapsedTime;
  time_t t1,t2;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  
  //
  //fill lhx data
  //
  std::cout<<"filling hlx/dip data"<<std::endl;  
  std::auto_ptr<lumi::DataPipe> ptr(lumi::DataPipeFactory::get()->create("Lumi2DB",lumidb));
  try{
     ptr->setAuthPath(authpath);
     ptr->setSource(lumifile);
     //ptr->setMode("beamintensity_only");
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
  if(ptr->getMode()!=std::string("beamintensity_only")){
     try{
	std::cout<<"fill out runsummary data"<<std::endl;
	std::auto_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create("CMSRunSummary2DB",lumidb));
	runptr->setSource(runinfodb);
	runptr->setAuthPath(authpath);
	startClock=clock();
	time(&t1);
	runptr->retrieveData(runnumber);
	time(&t2);
	endClock=clock();
     }catch(const lumi::nonCollisionException& er){
	std::cout<<"not a collision run, skip "<<std::endl;
	return 0;
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
	std::auto_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create("HLTConf2DB",lumidb));
	confptr->setSource(hltconfdb);
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
	std::cout<<"fill out trg data"<<std::endl;
	std::auto_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create("TRG2DB",lumidb));
	trgptr->setAuthPath(authpath);
	trgptr->setSource(trgdb);
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
	std::auto_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create("HLT2DB",lumidb));
	hltptr->setSource(runinfodb);
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
  }
  return 0;
}
