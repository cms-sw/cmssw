
#include "FWCore/CoreFramework/interface/EventProcessor.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/src/WorkerRegistry.h"
#include "FWCore/CoreFramework/interface/ScheduleBuilder.h"
#include "FWCore/CoreFramework/interface/ScheduleExecutor.h"
#include "FWCore/CoreFramework/src/InputServiceFactory.h"
#include "FWCore/CoreFramework/src/DebugMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeProcessPSet.h"

#include "PluginManager/PluginManager.h"

#include "FWCore/CoreFramework/interface/Timestamp.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/SourceFactory.h"
#include "FWCore/CoreFramework/interface/ModuleFactory.h"
#include "FWCore/CoreFramework/interface/EventRegistry.h"

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

  PathList tmpMakeSchedule(ParameterSet const& params_,
			const CommonParams& common,
			WorkerRegistry& reg_)
  {
    // can this fail? How?
    seal::PluginManager::get()->initialise();

    PathList workers;
    // get temporary single path
    StrVec tmp_path = params_.getVString("temporary_single_path");
    // call factory
    workers.push_back(WorkerList());

    StrVec::iterator pb(tmp_path.begin()),pe(tmp_path.end());
    for(;pb!=pe;++pb)
      {
	ParameterSet p = params_.getPSet(*pb);

	// reference return from registry is bad
	Worker* w = reg_.getWorker(p,
				   common.process_name_,
				   common.version_,
				   common.pass_);

	// this cannot happen right now - but is probably useful
	// if we change the return type from getWorker above
	if(w==0)
	  {
	    cerr << "Could not make worker type " 
		 << p.getString("module_type")
		 << " with label " 
		 << p.getString("module_label")
		 << endl;
	    throw runtime_error("EventProcessor could not make module");
	  }

	workers.front().push_back(w);
      }
    
    if(workers.empty())
	throw runtime_error("No workers have been placed into the schedule");

    return workers;
  }

  boost::shared_ptr<InputService> makeInput(ParameterSet const& params_,
					    const CommonParams& common)
  {
    // find single source
    ParameterSet main_input = params_.getPSet("main_input");
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
     vector<string> providers = params_.getVString("allesmodules");
     for( vector<string>::iterator itName = providers.begin();
          itName != providers.end();
          ++itName ) {
        ParameterSet providerPSet = params_.getPSet(*itName);
        ModuleFactory::get()->addTo( cp, 
                                     providerPSet, 
                                     common.process_name_, 
                                     common.version_, 
                                     common.pass_);
     }

     vector<string> sources = params_.getVString("allessources");
     for( vector<string>::iterator itName = sources.begin();
          itName != sources.end();
          ++itName ) {
        ParameterSet providerPSet = params_.getPSet(*itName);
        SourceFactory::get()->addTo( cp, 
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
    WorkerRegistry*         reg_;
    PathList                workers_;

    boost::shared_ptr<InputService> input_;
    ScheduleExecutor runner_;
    edm::eventsetup::EventSetupProvider cp_;    

    StrVec fillArgs(int argc, char* argv[]);
    string readFile(const StrVec& args);
  };
  
  FwkImpl::FwkImpl(int argc, char* argv[]) :
    args_(fillArgs(argc,argv)),
    configstring_(readFile(args_)),
    params_(makeProcessPSet(configstring_)),
    common_(
	    params_->getString("process_name"),
	    getVersion(), // this is not written for real yet
	    0), // how is this specifified? Where does it come from?
    reg_(WorkerRegistry::get()),
    //workers_(tmpMakeSchedule(*params_,common_,*reg_)),
    workers_(ScheduleBuilder(*params_).getPathList()),
    input_(makeInput(*params_,common_)),
    runner_(workers_)
  {
       fillEventSetupProvider(cp_, *params_, common_);
  }

  FwkImpl::FwkImpl(int argc, char* argv[], const string& config) :
    args_(fillArgs(argc,argv)),
    configstring_(config),
    params_(makeProcessPSet(configstring_)),
    common_(
	    params_->getString("process_name"),
	    getVersion(), // this is not written for real yet
	    0), // how is this specifified? Where does it come from?
    reg_(WorkerRegistry::get()),
    workers_(ScheduleBuilder(*params_).getPathList()), 
    //workers_(tmpMakeSchedule(*params_,common_,*reg_)),
    input_(makeInput(*params_,common_)),
    runner_(workers_)
  {
       fillEventSetupProvider(cp_, *params_, common_);
  }

  FwkImpl::FwkImpl(const string& config) :
    args_(),
    configstring_(config),
    params_(makeProcessPSet(configstring_)),
    common_(
	    params_->getString("process_name"),
	    getVersion(), // this is not written for real yet
	    0), // how is this specifified? Where does it come from?
    reg_(WorkerRegistry::get()),
    //workers_(tmpMakeSchedule(*params_,common_,*reg_)),
    workers_(ScheduleBuilder(*params_).getPathList()),
    input_(makeInput(*params_,common_)),
    runner_(workers_)
  {
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

    if(args.size()<3 || args[1]!=param_name )
      {
 	cerr << "No input file argument given.\n"
 	     << "Usage: " << args[0] << " --parameter-set pset_file_name"
 	     << endl;
 	throw runtime_error("No input pset given");
      }

    ifstream ist(args[2].c_str());
    
    if(!ist)
      {
 	cerr << "Input file " << args[2] << " could not be opened"
	     << endl;
 	throw runtime_error("pset input file could not be opened");
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
    //    boost::shared_ptr<DummyEventSetupRecordRetriever> pRetriever( new DummyEventSetupRecordRetriever );
    // cp.add( boost::shared_ptr<eventsetup::DataProxyProvider>(pRetriever) );
    
    // cp.add( boost::shared_ptr<eventsetup::EventSetupRecordIntervalFinder>(pRetriever) );


    bool runforever = numberToProcess==0;
    unsigned int eventcount=0;

    while(runforever || eventcount<numberToProcess )
      {
	++eventcount;
	FDEBUG(1) << eventcount << std::endl;
	auto_ptr<EventPrincipal> pep = input_->readEvent();
	if(pep.get()==0) break;
	edm::Timestamp ts(eventcount);
	EventSetup const& c = cp_.eventSetupForInstance(ts);

	runner_.runOneEvent(*pep.get(),c);
	EventRegistry::instance()->removeEvent(pep->ID());
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
