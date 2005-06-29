
#include "FWCore/CoreFramework/interface/EventProcessor.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/src/WorkerRegistry.h"
#include "FWCore/CoreFramework/interface/ScheduleBuilder.h"
#include "FWCore/CoreFramework/interface/ScheduleExecutor.h"
#include "FWCore/CoreFramework/src/InputServiceFactory.h"
#include "FWCore/CoreFramework/src/DebugMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"

#include "PluginManager/PluginManager.h"

#include "FWCore/CoreFramework/interface/Timestamp.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/SourceFactory.h"
#include "FWCore/CoreFramework/interface/ModuleFactory.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/EventRegistry.h"

#include "FWCore/FWUtilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>

using namespace std;


namespace edm {

  struct CommonParams
  {
    CommonParams():version_(),pass_() { }
    CommonParams(const string& name,unsigned long ver,unsigned long pass):
      process_name_(name),version_(ver),pass_(pass) { }

    string                  process_name_;
    unsigned long           version_;
    unsigned long           pass_;
  };

  typedef vector<string> StrVec;
  typedef list<string> StrList;
  typedef Worker* WorkerPtr;
  typedef list<WorkerPtr> WorkerList;
  typedef list<WorkerList> PathList;

  // temporary function because we do not know how to do this
  unsigned long getVersion() { return 0; }


  boost::shared_ptr<InputService> makeInput(ParameterSet const& params_,
					    const CommonParams& common)
  {
    // find single source
    ParameterSet main_input = params_.getParameter<ParameterSet>("main_input");
    InputServiceDescription isdesc(common.process_name_,common.pass_);

    boost::shared_ptr<InputService> input_
      (InputServiceFactory::get()->makeInputService(main_input,isdesc).release());
    
    return input_;
  }
  
  void fillEventSetupProvider(eventsetup::EventSetupProvider& cp,
                           ParameterSet const& params_,
                           const CommonParams& common)
  {
     using namespace std;
     using namespace edm::eventsetup;
     vector<string> providers = params_.getParameter<vector<string> >("all_esmodules");
     for(vector<string>::iterator itName = providers.begin();
          itName != providers.end();
          ++itName) {
        ParameterSet providerPSet = params_.getParameter<ParameterSet>(*itName);
        ModuleFactory::get()->addTo(cp, 
                                     providerPSet, 
                                     common.process_name_, 
                                     common.version_, 
                                     common.pass_);
     }

     vector<string> sources = params_.getParameter<vector<string> >("all_essources");
     for(vector<string>::iterator itName = sources.begin();
          itName != sources.end();
          ++itName) {
        ParameterSet providerPSet = params_.getParameter<ParameterSet>(*itName);
        SourceFactory::get()->addTo(cp, 
                                     providerPSet, 
                                     common.process_name_, 
                                     common.version_, 
                                     common.pass_);
     }
  }
  // right now we only support a pset string from constructor or
  // pset read from file

  struct FwkImpl
  {
    FwkImpl(int argc, char* argv[]);
    FwkImpl(int argc, char* argv[], const string& config);
    explicit FwkImpl(const string& config);

    EventProcessor::StatusCode run(unsigned long numberToProcess);
   
    StrVec                  args_;
    string                  configstring_;
    boost::shared_ptr<ParameterSet> params_;
    CommonParams            common_;
    WorkerRegistry          reg_;
    PathList                workers_;

    boost::shared_ptr<InputService> input_;
    ScheduleExecutor runner_;
    edm::eventsetup::EventSetupProvider cp_;    

    StrVec fillArgs(int argc, char* argv[]);
    string readFile(const StrVec& args);
  };
  
  FwkImpl::FwkImpl(int argc, char* argv[]) :
    args_(fillArgs(argc,argv)),
    configstring_(readFile(args_))
  {
    
    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    ScheduleBuilder sbuilder= ScheduleBuilder(*params_,&reg_);
    
    workers_= (sbuilder.getPathList());
    input_= makeInput(*params_,common_);
    runner_ = ScheduleExecutor(workers_);
    
    fillEventSetupProvider(cp_, *params_, common_);
  }

  FwkImpl::FwkImpl(int argc, char* argv[], const string& config) :
    args_(fillArgs(argc,argv)),
    configstring_(config){
    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    ScheduleBuilder sbuilder= ScheduleBuilder(*params_,&reg_);
    
    workers_= (sbuilder.getPathList());
    input_= makeInput(*params_,common_);
    runner_ = ScheduleExecutor(workers_);
    fillEventSetupProvider(cp_, *params_, common_);

  }

  FwkImpl::FwkImpl(const string& config) :
    args_(),
    configstring_(config){

    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    ScheduleBuilder sbuilder= ScheduleBuilder(*params_,&reg_);
    
    workers_= (sbuilder.getPathList());
    input_= makeInput(*params_,common_);
    runner_ = ScheduleExecutor(workers_);
    
    FDEBUG(2) << params_->toString() << std::endl;
  }

  StrVec FwkImpl::fillArgs(int argc, char* argv[])
  {
    StrVec args;
    copy(&argv[0],&argv[argc],back_inserter(args));
    return args;
  }
  
  string FwkImpl::readFile(const StrVec& args)
  {
    string param_name("--parameter-set");

    if(args.size()<3 || args[1]!=param_name)
      {
 	throw edm::Exception(errors::Configuration,"MissingArgument")
	  << "No input file argument given (pset name).\n"
	  << "Usage: " << args[0] << " --parameter-set pset_file_name"
	  << endl;
      }

    ifstream ist(args[2].c_str());
    
    if(!ist)
      {
 	throw edm::Exception(errors::Configuration,"OpenFile")
	  << "pset input file could not be opened\n"
	  << "Input file " << args[2] << " could not be opened"
	  << endl;
      }

    string configstring;
    string line;

    while(std::getline(ist,line)) { configstring+=line; configstring+="\n"; }

    FDEBUG(2) << "configuration:\n"
	      << configstring << std::endl;
    return configstring;
  }

  // notice that exception catching is missing...

  EventProcessor::StatusCode
  FwkImpl::run(unsigned long numberToProcess)
  {

    // Setup the EventSetup
    //    boost::shared_ptr<DummyEventSetupRecordRetriever> pRetriever(new DummyEventSetupRecordRetriever);
    // cp.add(boost::shared_ptr<eventsetup::DataProxyProvider>(pRetriever));
    
    // cp.add(boost::shared_ptr<eventsetup::EventSetupRecordIntervalFinder>(pRetriever));


    bool runforever = numberToProcess==0;
    unsigned int eventcount=0;

    while(runforever || eventcount<numberToProcess)
      {
	++eventcount;
	FDEBUG(1) << eventcount << std::endl;
	auto_ptr<EventPrincipal> pep = input_->readEvent();
	if(pep.get()==0) break;
	edm::Timestamp ts(eventcount);
	EventSetup const& c = cp_.eventSetupForInstance(ts);

	EventRegistry::instance()->addEvent(pep->id(), pep.get());
	runner_.runOneEvent(*pep.get(),c);
	EventRegistry::instance()->removeEvent(pep->id());
      }

    return 0;
  }


  // ------------------------------------------

  EventProcessor::EventProcessor(int argc, char* argv[]):
    impl_(new FwkImpl(argc,argv))
  {
  } 
  
  EventProcessor::EventProcessor(const string& config):
    impl_(new FwkImpl(config))
  {
  } 
  
  EventProcessor::EventProcessor(int argc, char* argv[], const string& config):
    impl_(new FwkImpl(argc,argv,config))
  {
  }

  EventProcessor::~EventProcessor()
  {
    delete impl_;
  }

  EventProcessor::StatusCode
  EventProcessor::run(unsigned long numberToProcess)
  {
    return impl_->run(numberToProcess);
  }
}
