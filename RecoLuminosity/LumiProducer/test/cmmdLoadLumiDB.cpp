#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/ISchema.h"
#include "CoralBase/Exception.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <cstdio>

#include <boost/program_options.hpp>

int main(int argc, char** argv){
  std::string lumipluginName("Lumi2DB");
  std::string trgpluginName("TRGScalers2DB");
  std::string wbmpluginName("TRGWBM2DB");
  std::string hltpluginName("HLTV32DB");
  std::string hltconfpluginName("HLTConf2DB");
  std::string runsummarypluginName("CMSRunSummary2DB");
  std::string defaultRuninfoConnect("oracle://cms_omds_lb/CMS_RUNINFO");
  std::string defaultTRGConnect("oracle://cms_omds_lb/CMS_GT_MON");
  std::string defaultWBMConnect("oracle://cms_omds_lb/CMS_WBM");
  std::string defaultHLTConfConnect("oracle://cms_omds_lb/CMS_HLT");
  
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmmdLoadLumiDB [options] \n");
  visible.add_options()
    ("runnumber,r",boost::program_options::value<unsigned int>(),"runnumber(required)")
    ("destConnect,c",boost::program_options::value<std::string>(),"destionation connection (required)")
    ("lumipath,L",boost::program_options::value<std::string>(),"path to lumi data file(required)")
    ("authpath,P",boost::program_options::value<std::string>(),"path to authentication xml(default .)")
    ("runinfodb,R",boost::program_options::value<std::string>(),"connect to runinfodb")
    ("confdb,C",boost::program_options::value<std::string>(),"connect to hltconfdb")
    ("trgdb,T",boost::program_options::value<std::string>(),"connect to trgdb")
    ("wbmdb,W",boost::program_options::value<std::string>(),"connect to wbmdb")    
    ("configFile,f",boost::program_options::value<std::string>(),"configuration file(optional)")
    ("norm,n",boost::program_options::value<float>(),"norm factor(optional)")
    ("patchcomment,a",boost::program_options::value<std::string>(),"patchcomment(optional)")
    ("without-lumi","exclude lumi loading")
    ("without-trg","exclude trg loading")
    ("without-hlt","exclude hlt loading")
    ("without-runsummary","exclude runsummary loading")
    ("without-hconf","exclude hltconf loading")
    ("use-wbm","use wbmdb for trigger info")
    ("collision-only","load collision/physics run only")
    ("schemav2-only","load only to schemav2")
    ("nocheckingstablebeam","do not require any presence of stable beam")
    ("novalidate","do not validate lumi data")
    ("dryrun","dryrun print parameter only")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  std::string configuration_filename;
  unsigned int runnumber;
  std::string authpath(".");
  std::string destconnect;
  std::string lumipath;
  std::string runinfodb=defaultRuninfoConnect;
  std::string trgdb=defaultTRGConnect;
  std::string wbmdb=defaultWBMConnect;
  std::string hltconfdb=defaultHLTConfConnect;
  std::string patchcomment("");
  float norm=0.0;
  bool without_lumi=false;
  bool without_trg=false;
  bool without_hlt=false;
  bool without_runsummary=false;
  bool without_hltconf=false;
  bool use_wbm=false;
  bool debug=false;
  bool novalidate=false;
  bool collision_only=false;
  bool schemav2_only=false;
  bool nocheckingstablebeam=false;
  bool dryrun=false;
  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( vm.count("configFile") ){
      configuration_filename=vm["configFile"].as<std::string>();
      if (! configuration_filename.empty()){
	std::fstream configuration_file;
	configuration_file.open(configuration_filename.c_str(), std::fstream::in);
	boost::program_options::store(boost::program_options::parse_config_file(configuration_file,desc), vm);
	configuration_file.close();
      }
    }
    if(!vm.count("runnumber")){
      std::cerr <<"[Error] runnumber[r] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      runnumber=vm["runnumber"].as<unsigned int>();
    }
    if(!vm.count("destConnect")){
      std::cerr <<"[Error] no destConnect[c] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      destconnect=vm["destConnect"].as<std::string>();
    }
    if( vm.count("lumipath") ){
      lumipath=vm["lumipath"].as<std::string>();
    }
    if( vm.count("authpath") ){
      authpath=vm["authpath"].as<std::string>();
    }
    if( vm.count("runinfodb") ){
      runinfodb=vm["runinfodb"].as<std::string>();
    }
    if( vm.count("confdb") ){
      hltconfdb=vm["confdb"].as<std::string>();
    }
     if( vm.count("trgdb") ){
      trgdb=vm["trgdb"].as<std::string>();
    }
    if( vm.count("wbmdb") ){
      wbmdb=vm["wbmdb"].as<std::string>();
    }
    if( vm.count("patchcomment") ){
      patchcomment=vm["patchcomment"].as<std::string>();
    }
    if( vm.count("norm") ){
      norm=vm["norm"].as<float>();
    }
    if(vm.count("without-lumi") ){
      without_lumi=true;
    }
    if(vm.count("without-trg") ){
      without_trg=true;
    }
    if(vm.count("without-hlt") ){
      without_hlt=true;
    }
    if(vm.count("without-runsummary") ){
      without_runsummary=true;
    }
    if(vm.count("without-hconf") ){
      without_hltconf=true;
    }
    if(vm.count("use-wbm") ){
      use_wbm=true;
    }
    if(vm.count("novalidate") ){
      novalidate=true;
    }
    if(vm.count("collision-only") ){
       collision_only=true;
    }
    if(vm.count("schemav2-only") ){
      schemav2_only=true;
    }
    if(vm.count("nocheckingstablebeam") ){
      nocheckingstablebeam=true;
    }
    if(vm.count("dryrun") ){
      dryrun=true;
    }
    if(vm.count("debug") ){
      debug=true;
    }
    if(!without_lumi && lumipath.size()==0){
      std::cerr <<"[Error] lumipath[L] option is required \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  if(debug){
    std::cout<<"runnumber "<<runnumber<<std::endl;
    std::cout<<"destConnect "<<destconnect<<std::endl;
    std::cout<<"authpath "<<authpath<<std::endl;
    std::cout<<"lumipath "<<lumipath<<std::endl;
    std::cout<<"runinfodb "<<runinfodb<<std::endl;
    std::cout<<"hltconfdb "<<hltconfdb<<std::endl;
    std::cout<<"trgdb "<<trgdb<<std::endl;
    std::cout<<"wbmdb "<<wbmdb<<std::endl;
    std::string answer;
    (without_lumi)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"loading lumi? "<<answer<<std::endl;
    (without_trg)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"loading trg? "<<answer<<std::endl;
    (without_hlt)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"loading hlt? "<<answer<<std::endl;
    (without_runsummary)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"loading runsummary? "<<answer<<std::endl;
    (without_hltconf)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"loading hltconf? "<<answer<<std::endl;
    (use_wbm)?(answer=std::string("Yes")):(answer=std::string("No"));
    std::cout<<"using wbm as trigger source? "<<answer<<std::endl;    
    (novalidate)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"validate data? "<<answer<<std::endl;
    (collision_only)?(answer=std::string("Yes")):(answer=std::string("No"));
    std::cout<<"collision only ? "<<answer<<std::endl;
    (schemav2_only)?(answer=std::string("Yes")):(answer=std::string("No"));
    std::cout<<"schemav2 only ? "<<answer<<std::endl;
    (nocheckingstablebeam)?(answer=std::string("No")):(answer=std::string("Yes"));
    std::cout<<"checking stablebeam ? "<<answer<<std::endl;
    (dryrun)?(answer=std::string("Yes")):(answer=std::string("No"));    
    std::cout<<"dryrun? "<<answer<<std::endl;
  }
  if(dryrun) return 0;
  clock_t startClock, endClock;
  double elapsedTime=0.0;
  time_t t1,t2;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::cout<<"=========="<<std::endl;
  unsigned int lumidataid=0;
  unsigned int trgdataid=0;
  unsigned int hltdataid=0;
  if(!without_lumi){
    std::cout<<"Loading lumi from "<<lumipath<<" to "<< destconnect <<" run "<<runnumber<<" with "<<lumipluginName<<std::endl;
    try{
      std::unique_ptr<lumi::DataPipe> lumiptr(lumi::DataPipeFactory::get()->create(lumipluginName,destconnect));
      lumiptr->setAuthPath(authpath);
      lumiptr->setSource(lumipath);
      if(novalidate){
	lumiptr->setNoValidate();
      }
      if(nocheckingstablebeam){
	lumiptr->setNoCheckingStableBeam();
      }
      if(norm!=0.0){
	std::cout<<"applying norm "<<norm<<std::endl;
	lumiptr->setNorm(norm);
	
      }
      //lumiptr->setMode("beamintensity_only");
      if(!schemav2_only){
	 lumiptr->setMode("loadoldschema");	 
      }
      startClock=clock();
      time(&t1);
      lumidataid=lumiptr->retrieveData(runnumber);
      time(&t2);
      endClock=clock();
      lumiptr.release();
    }catch(const lumi::invalidDataException& er){
      //if (novalidate){
      //	std::cout<<"\t [WARNING]lumi data for this run is not valid, load anyway\n";
      //	std::cout<<"\t"<<er.what()<<std::endl;
      //}else{
      std::cout<<"\t [ERROR]lumi data for this run is not valid, stop loading immediately";
      std::cout<<"\t"<<er.what()<<std::endl;
      return 1;
      //}
    }catch(const lumi::noStableBeamException& er){
      std::cout<<"\t [ERROR]lumi data for this run has no stable beam, stop loading immediately";
      std::cout<<"\t"<<er.what()<<std::endl;
      return 1;
    }catch(const coral::Exception& er){
      std::cout<<"\t Database error "<<er.what()<<std::endl;
      throw;
    }catch(...){
      std::cout<<"\tproblem in loading lumi  "<<runnumber<<" SKIP "<<std::endl;
      throw;
    }
    printf("\tElaspsed time %fs\n",difftime(t2,t1));
    elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
    std::cout<<"\tCPU Time taken in seconds : "<<elapsedTime<<std::endl;
  }
  if(!without_runsummary){
    std::cout<<"Loading runsummary from "<<runinfodb<<" to "<<destconnect <<" run "<<runnumber<<" with "<<runsummarypluginName<<std::endl;
     try{
	std::unique_ptr<lumi::DataPipe> runptr(lumi::DataPipeFactory::get()->create(runsummarypluginName,destconnect));
	runptr->setSource(runinfodb);
	runptr->setAuthPath(authpath);
	if(!collision_only){
	   runptr->setNoValidate();
	}
	startClock=clock();
	time(&t1);
	runptr->retrieveData(runnumber);
	time(&t2);
	endClock=clock();
	runptr.release();
     }catch(const lumi::invalidDataException& er){   
	std::cout<<"\t [ERROR], run is not collision run, stop loading immediately";
	std::cout<<"\t"<<er.what()<<std::endl;	
	return 1;
     }catch(const coral::Exception& er){
	std::cout<<"\t Database error "<<er.what()<<std::endl;
	throw;
     }catch(...){
	std::cout<<"\tproblem in loading runsummary "<<runnumber<<" SKIP "<<std::endl;
	throw;
     }
     printf("\tElaspsed time %fs\n",difftime(t2,t1));
     elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
     std::cout<<"\tCPU Time taken in seconds : "<<elapsedTime<<std::endl;
  }
  if(!without_hltconf){
    try{
      std::cout<<"Loading hlt conf from "<<hltconfdb<<" to "<<destconnect <<" run "<<runnumber<<" with "<<hltconfpluginName<<std::endl;
      std::unique_ptr<lumi::DataPipe> confptr(lumi::DataPipeFactory::get()->create(hltconfpluginName,destconnect));
      confptr->setSource(hltconfdb);
      confptr->setAuthPath(authpath);
      startClock=clock();
      time(&t1);
      confptr->retrieveData(runnumber);
      time(&t2);
      endClock=clock();
      confptr.release();
    }catch(const coral::Exception& er){
      std::cout<<"\t Database error "<<er.what()<<std::endl;
      throw;
    }catch(...){
      std::cout<<"\tproblem in loading hltconf "<<runnumber<<" SKIP "<<std::endl;
      throw;
    }
    printf("\tElaspsed time %fs\n",difftime(t2,t1));
    elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
    std::cout<<"\tCPU Time taken in seconds : "<<elapsedTime<<std::endl;
  }
  if(!without_trg){
    try{
      if(!use_wbm){
	std::cout<<"Loading trg from GT "<<trgdb<<" to "<<destconnect <<" run "<<runnumber<<" with "<<trgpluginName<<std::endl;
	std::unique_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create(trgpluginName,destconnect));
	trgptr->setAuthPath(authpath);
	trgptr->setSource(trgdb);
	if(!schemav2_only){
	   trgptr->setMode("loadoldschema");	 
	}
	startClock=clock();
	time(&t1);
	trgdataid=trgptr->retrieveData(runnumber);
	time(&t2);
	endClock=clock();
	trgptr.release();
      }else{
	std::cout<<"Loading trg from WBM "<<wbmdb<<" to "<<destconnect <<" run "<<runnumber<<" with "<<wbmpluginName<<std::endl;
	std::unique_ptr<lumi::DataPipe> trgptr(lumi::DataPipeFactory::get()->create(wbmpluginName,destconnect));
	trgptr->setAuthPath(authpath);
	trgptr->setSource(wbmdb);
	startClock=clock();
	time(&t1);
	trgdataid=trgptr->retrieveData(runnumber);
	time(&t2);
	endClock=clock();
	trgptr.release();
      }
    }catch(const coral::Exception& er){
      std::cout<<"\t Database error "<<er.what()<<std::endl;
      throw;
    }catch(...){
      std::cout<<"\tproblem in loading trigger "<<runnumber<<" SKIP "<<std::endl;
      throw;
    }
    printf("\tElaspsed time %fs\n",difftime(t2,t1));
    elapsedTime=((double) (endClock - startClock)) / CLOCKS_PER_SEC;
    std::cout<<"\tCPU Time taken in seconds : "<<elapsedTime<<std::endl;
  }
  if(!without_hlt){
    std::cout<<"Loading hlt from Runinfo "<<runinfodb <<" to "<<destconnect<<" run "<<runnumber<<" with "<<hltpluginName<<std::endl;
    try{
      std::unique_ptr<lumi::DataPipe> hltptr(lumi::DataPipeFactory::get()->create(hltpluginName,destconnect));
      hltptr->setSource(runinfodb);
      hltptr->setAuthPath(authpath);
      if(!schemav2_only){
	 hltptr->setMode("loadoldschema");	 
      }
      startClock=clock();
      time(&t1);
      hltdataid=hltptr->retrieveData(runnumber);
      time(&t2);
      endClock=clock();
      hltptr.release();
    }catch(const coral::Exception& er){
      std::cout<<"\t Database error "<<er.what()<<std::endl;
      throw;
    }catch(...){
      std::cout<<"\tproblem in loading hlt "<<runnumber<<" SKIP "<<std::endl;
      throw;
    }
    printf("\tElaspsed time %fs\n",difftime(t2,t1));
    std::cout<<"\tCPU Time taken in seconds : "<<elapsedTime<<std::endl;
  }

  //
  //register ids and patchcomment
  //
  coral::ConnectionService* svc=new coral::ConnectionService;
  lumi::DBConfig dbconf(*svc);
  if(!authpath.empty()){
    dbconf.setAuthentication(authpath);
  }
  coral::ISessionProxy* session=svc->connect(destconnect,coral::Update);
  coral::ITypeConverter& lumitpc=session->typeConverter();
  lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
  lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
  lumitpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");
  coral::ITransaction& transaction=session->transaction();
  transaction.start(false);
  lumi::RevisionDML revisionDML;
  unsigned long long currenttagiid=0;
  try{
    currenttagiid=revisionDML.addRunToCurrentHFDataTag(session->nominalSchema(),runnumber,lumidataid,trgdataid,hltdataid,patchcomment);
  }catch(const lumi::duplicateRunInDataTagException& ){
    std::cout<<"\t[ERROR] run already registered in the current tag, create a new tag for the patches"<<std::endl;
    transaction.rollback();
    delete session;
    return 0;
  }catch(const coral::Exception& er){
    std::cout<<"\t[ERROR] Database error "<<er.what()<<std::endl;
    throw;
  }catch(...){
    std::cout<<"\t[ERROR] problem in registering run to tag "<<runnumber<<" SKIP "<<std::endl;
    throw;
  }
  transaction.commit();
  std::cout<<"register run to current data tag id="<<currenttagiid<<" run "<< runnumber<<" dataid: "<<lumidataid<<"-"<<trgdataid<<"-"<<hltdataid<<" "<<patchcomment<<std::endl;
  delete session;
  return 0;

}
