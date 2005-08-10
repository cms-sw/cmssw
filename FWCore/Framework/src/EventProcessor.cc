
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/interface/ScheduleBuilder.h"
#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/src/InputServiceFactory.h"
#include "FWCore/Framework/src/DebugMacros.h"
#include "FWCore/Framework/interface/Actions.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"

#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventRegistry.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"

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
      processName_(name),version_(ver),pass_(pass) { }

    string                  processName_;
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
					    const CommonParams& common,
					    ProductRegistry& preg)
  {
    // find single source
    ParameterSet main_input = params_.getParameter<ParameterSet>("main_input");
    InputServiceDescription isdesc(common.processName_,common.pass_,preg);

    boost::shared_ptr<InputService> input_
      (InputServiceFactory::get()->makeInputService(main_input, isdesc).release());
    
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
                                     common.processName_, 
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
                                     common.processName_, 
                                     common.version_, 
                                     common.pass_);
     }
  }

  // -------------------------------------------------------------------
  //              implementation class
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
    WorkerRegistry          wreg_;
    ProductRegistry         preg_;
    PathList                workers_;

    boost::shared_ptr<InputService> input_;
    ScheduleExecutor runner_;
    edm::eventsetup::EventSetupProvider esp_;    

    bool emittedBeginJob_;
    ActionTable act_table_;
    
    StrVec fillArgs(int argc, char* argv[]);
    string readFile(const StrVec& args);
  };

  // ---------------------------------------------------------------
  
  FwkImpl::FwkImpl(int argc, char* argv[]) :
    args_(fillArgs(argc,argv)),
    configstring_(readFile(args_)),
    emittedBeginJob_(false)
  {
    
    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    // this organization leads to unnecessary copies being made
    act_table_ = ActionTable(*params_);
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    input_= makeInput(*params_, common_, preg_);
    ScheduleBuilder sbuilder= 
      ScheduleBuilder(*params_, wreg_, preg_, act_table_);
    
    workers_= (sbuilder.getPathList());
    runner_ = ScheduleExecutor(workers_, act_table_);
    
    fillEventSetupProvider(esp_, *params_, common_);
  }
  
  FwkImpl::FwkImpl(int argc, char* argv[], const string& config) :
    args_(fillArgs(argc,argv)),
    configstring_(config),
    emittedBeginJob_(false) {
    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    act_table_ = ActionTable(*params_);
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    input_= makeInput(*params_, common_, preg_);
    ScheduleBuilder sbuilder= 
      ScheduleBuilder(*params_, wreg_, preg_, act_table_);
    
    workers_= (sbuilder.getPathList());
    runner_ = ScheduleExecutor(workers_,act_table_);
    fillEventSetupProvider(esp_, *params_, common_);

  }

  FwkImpl::FwkImpl(const string& config) :
    args_(),
    configstring_(config),
    emittedBeginJob_(false) {

    ProcessPSetBuilder builder(configstring_);
    params_ = builder.getProcessPSet();
    act_table_ = ActionTable(*params_);
    common_ = 
      CommonParams((*params_).getParameter<string>("process_name"),
		   getVersion(), // this is not written for real yet
		   0); // how is this specifified? Where does it come from?
 
    input_= makeInput(*params_, common_, preg_);
    ScheduleBuilder sbuilder= 
      ScheduleBuilder(*params_, wreg_, preg_, act_table_);
    
    workers_= (sbuilder.getPathList());
    runner_ = ScheduleExecutor(workers_, act_table_);
    
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

  //need a wrapper to let me 'copy' references to EventSetup
  namespace eventprocessor {
     struct ESRefWrapper {
        EventSetup const & es_;
        ESRefWrapper(EventSetup const &iES) : es_(iES) {}
        operator const EventSetup&() { return es_; }
     };
  }
  using eventprocessor::ESRefWrapper;
  
  EventProcessor::StatusCode
  FwkImpl::run(unsigned long numberToProcess)
  {

    bool runforever = numberToProcess==0;
    unsigned int eventcount=0;

    //NOTE:  This implementation assumes 'Job' means one call the EventProcessor::run
    // If it really means once per 'application' then this code will have to be changed.
    // Also have to deal with case where have 'run' then new Module added and do 'run'
    // again.  In that case the newly added Module needs its 'beginJob' to be called.
    EventSetup const& es = esp_.eventSetupForInstance(edm::IOVSyncValue::beginOfTime());
    PathList::iterator itWorkerList = workers_.begin();
    PathList::iterator itEnd = workers_.end();
    ESRefWrapper wrapper( es );
    for( ; itWorkerList != itEnd; ++itEnd ) {
       std::for_each( itWorkerList->begin(), itWorkerList->end(), 
                      boost::bind( boost::mem_fn(&Worker::beginJob), _1, wrapper) );
    }

    while(runforever || eventcount<numberToProcess)
      {
	++eventcount;
	FDEBUG(1) << eventcount << std::endl;
	auto_ptr<EventPrincipal> pep = input_->readEvent();
	if(pep.get()==0) break;
	edm::IOVSyncValue ts(pep->id(), pep->time());
	EventSetup const& es = esp_.eventSetupForInstance(ts);

	try
	  {
	    EventRegistry::Operate oper(pep->id(),pep.get());
	    runner_.runOneEvent(*pep.get(),es);
	  }
	catch(cms::Exception& e)
	  {
	    actions::ActionCodes code = act_table_.find(e.rootCause());
	    if(code==actions::IgnoreCompletely)
	      {
		// change to error logger!
		cerr << "Ignoring exception from Event ID=" << pep->id()
		     << ", message:\n" << e.what()
		     << endl;
		continue;
	      }
	    else if(code==actions::SkipEvent)
	      {
		cerr << "Skipping Event ID=" << pep->id()
		     << ", message:\n" << e.what()
		     << endl;
		continue;
	      }
	    else
	      throw edm::Exception(errors::EventProcessorFailure,
				   "EventProcessingStopped",e);
	  }
      }

    //NOTE: this is not done if an exception is thrown in the above loop.
    // This was done intentionally.
    {
       PathList::const_iterator itWorkerList = workers_.begin();
       PathList::const_iterator itEnd = workers_.end();
       for( ; itWorkerList != itEnd; ++itEnd ) {
          std::for_each( itWorkerList->begin(), itWorkerList->end(), 
                         boost::mem_fn(&Worker::endJob) );
       }
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
