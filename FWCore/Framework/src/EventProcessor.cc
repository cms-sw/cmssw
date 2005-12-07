
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/interface/ScheduleBuilder.h"
#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
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
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/EDProduct/interface/EDProductGetter.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

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


  boost::shared_ptr<InputSource> makeInput(ParameterSet const& params_,
					    const CommonParams& common,
					    ProductRegistry& preg)
  {
    // find single source
    bool sourceSpecified = false;
    try {
      ParameterSet main_input = params_.getParameter<ParameterSet>("@main_input");
      sourceSpecified = true;
      InputSourceDescription isdesc(common.processName_,common.pass_,preg);

      boost::shared_ptr<InputSource> input_
      (InputSourceFactory::get()->makeInputSource(main_input, isdesc).release());
    
      return input_;
    } catch(const edm::Exception& iException) {
      if(sourceSpecified == false && errors::Configuration == iException.categoryCode()) {
        throw edm::Exception(errors::Configuration, "NoInputSource")
        <<"No main input source found in configuration.  Please add an input source via 'source = ...' in the configuration file.\n";
      } else {
        throw;
      }
    }
    return boost::shared_ptr<InputSource>();
  }
  
  void fillEventSetupProvider(eventsetup::EventSetupProvider& cp,
                           ParameterSet const& params_,
                           const CommonParams& common)
  {
     using namespace std;
     using namespace edm::eventsetup;
     vector<string> providers = params_.getParameter<vector<string> >("@all_esmodules");
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

     vector<string> sources = params_.getParameter<vector<string> >("@all_essources");
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
    FwkImpl(int argc, char* argv[],
            const ServiceToken& = ServiceToken(),
            serviceregistry::ServiceLegacy=serviceregistry::kOverlapIsError);
    FwkImpl(int argc, char* argv[], const string& config,
            const ServiceToken& = ServiceToken(),
            serviceregistry::ServiceLegacy=serviceregistry::kOverlapIsError);
    explicit FwkImpl(const string& config,
                     const ServiceToken& = ServiceToken(),
                     serviceregistry::ServiceLegacy=serviceregistry::kOverlapIsError);

    EventProcessor::StatusCode run(unsigned long numberToProcess);
    void beginJob();
    bool endJob();
    
    StrVec                  args_;
    string                  configstring_;
    boost::shared_ptr<ParameterSet> params_;
    CommonParams            common_;
    WorkerRegistry          wreg_;
    SignallingProductRegistry preg_;
    PathList                workers_;

    ActivityRegistry activityRegistry_;
    ServiceToken serviceToken_;
    
    boost::shared_ptr<InputSource> input_;
    std::auto_ptr<ScheduleExecutor> runner_;
    edm::eventsetup::EventSetupProvider esp_;    

    bool emittedBeginJob_;
    ActionTable act_table_;
    
    StrVec fillArgs(int argc, char* argv[]);
    string readFile(const StrVec& args);
    
    
    private:
       void initialize(const ServiceToken& iToken, serviceregistry::ServiceLegacy iLegacy);

  };

  // ---------------------------------------------------------------
  void FwkImpl::initialize(const ServiceToken& iToken, serviceregistry::ServiceLegacy iLegacy)
  {    
     ProcessPSetBuilder builder(configstring_);

     //create the services
     boost::shared_ptr< std::vector<edm::ParameterSet> > pServiceSets(builder.getServicesPSets());
     //NOTE: FIX WHEN POOL BUG FIXED
     // we force in the LoadAllDictionaries service in order to work around a bug in POOL
     {
        edm::ParameterSet ps;
        std::string type("LoadAllDictionaries");
        ps.addParameter("@service_type",type);
        pServiceSets->push_back( ps );
     }
     serviceToken_ = ServiceRegistry::createSet(*pServiceSets,
                                                iToken,iLegacy);
     serviceToken_.connectTo(activityRegistry_);
     
     //add the ProductRegistry as a service ONLY for the construction phase
     boost::shared_ptr<serviceregistry::ServiceWrapper<ConstProductRegistry> > 
        reg(new serviceregistry::ServiceWrapper<ConstProductRegistry>( 
                   std::auto_ptr<ConstProductRegistry>(new ConstProductRegistry(preg_))));
     ServiceToken tempToken( ServiceRegistry::createContaining(reg, serviceToken_, serviceregistry::kOverlapIsError));
     //make the services available
     ServiceRegistry::Operate operate(tempToken);
     
     params_ = builder.getProcessPSet();
     act_table_ = ActionTable(*params_);
     common_ = 
        CommonParams((*params_).getParameter<string>("@process_name"),
                     getVersion(), // this is not written for real yet
                     0); // how is this specifified? Where does it come from?
     
     input_= makeInput(*params_, common_, preg_);
     ScheduleBuilder sbuilder(*params_, wreg_, preg_, act_table_);
     
     workers_= (sbuilder.getPathList());
     runner_ = std::auto_ptr<ScheduleExecutor>(new ScheduleExecutor(workers_,act_table_));
     runner_->preModuleSignal.connect(activityRegistry_.preModuleSignal_);
     runner_->postModuleSignal.connect(activityRegistry_.postModuleSignal_);
     
     
     fillEventSetupProvider(esp_, *params_, common_);
  }
  
  FwkImpl::FwkImpl(int argc, char* argv[],
     const ServiceToken& iToken, serviceregistry::ServiceLegacy iLegacy):
    args_(fillArgs(argc,argv)),
    configstring_(readFile(args_)),
    emittedBeginJob_(false)
  {
    initialize(iToken,iLegacy);
  }
  
  FwkImpl::FwkImpl(int argc, char* argv[], const string& config,
                   const ServiceToken& iToken, serviceregistry::ServiceLegacy iLegacy):
    args_(fillArgs(argc,argv)),
    configstring_(config),
    emittedBeginJob_(false) {
    initialize(iToken,iLegacy);
  }

  FwkImpl::FwkImpl(const string& config,
                   const ServiceToken& iToken, serviceregistry::ServiceLegacy iLegacy):
    args_(),
    configstring_(config),
    emittedBeginJob_(false) {
    initialize(iToken,iLegacy);
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

  void
  FwkImpl::beginJob() 
  {
     //make the services available
     ServiceRegistry::Operate operate(serviceToken_);

     if(! emittedBeginJob_) {
        //NOTE:  This implementation assumes 'Job' means one call the EventProcessor::run
        // If it really means once per 'application' then this code will have to be changed.
        // Also have to deal with case where have 'run' then new Module added and do 'run'
        // again.  In that case the newly added Module needs its 'beginJob' to be called.
        EventSetup const& es = esp_.eventSetupForInstance(edm::IOVSyncValue::beginOfTime());
        PathList::iterator itWorkerList = workers_.begin();
        PathList::iterator itEnd = workers_.end();
        ESRefWrapper wrapper(es);
        
        for(; itWorkerList != itEnd; ++itEnd) {
           std::for_each(itWorkerList->begin(), itWorkerList->end(), 
                         boost::bind(boost::mem_fn(&Worker::beginJob), _1, wrapper));
        }
        emittedBeginJob_ = true;
        activityRegistry_.postBeginJobSignal_();
     }
  }

  bool
  FwkImpl::endJob() 
  {
     //make the services available
     ServiceRegistry::Operate operate(serviceToken_);
     
     bool returnValue = true;
     PathList::const_iterator itWorkerList = workers_.begin();
     PathList::const_iterator itEnd = workers_.end();
     for(; itWorkerList != itEnd; ++itEnd) {
        for(WorkerList::const_iterator itWorker = itWorkerList->begin();
            itWorker != itWorkerList->end();
            ++itWorker) {
           try {
              (*itWorker)->endJob();
           } catch(cms::Exception& iException) {
              cerr<<"Caught cms::Exception in endJob: "<< iException.what()<<endl;
              returnValue = false;
           } catch(std::exception& iException) {
              cerr<<"Caught std::exception in endJob: "<< iException.what()<<endl;
              cerr<<endl;
              returnValue = false;
           } catch(...) {
              cerr<<"Caught unknown exception in endJob."<<endl;
              returnValue = false;
           }
        }
     }     
     
     activityRegistry_.postEndJobSignal_();
     return returnValue;
  }
  
  EventProcessor::StatusCode
  FwkImpl::run(unsigned long numberToProcess)
  {
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    bool runforever = numberToProcess==0;
    unsigned int eventcount=0;

    //make sure this was called
    beginJob();

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
            ModuleDescription dummy;
            {
              activityRegistry_.preProcessEventSignal_(pep->id(),pep->time());
            }
	    runner_->runOneEvent(*pep.get(),es);
            {
              activityRegistry_.postProcessEventSignal_(Event(*pep.get(),dummy) , es);
            }
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

    return 0;
  }


  // ------------------------------------------

  static void connectSigs(EventProcessor* iEP, FwkImpl* iImpl) {
     //When the FwkImpl signals are given, pass them to the appropriate EventProcessor
     // signals so that the outside world can see the signal
     iImpl->activityRegistry_.preProcessEventSignal_.connect(iEP->preProcessEventSignal);
     iImpl->activityRegistry_.postProcessEventSignal_.connect(iEP->postProcessEventSignal);
  }
  EventProcessor::EventProcessor(int argc, char* argv[]):
    impl_(new FwkImpl(argc,argv))
  {
       connectSigs(this, impl_);
  } 
  
  EventProcessor::EventProcessor(const string& config):
    impl_(new FwkImpl(config))
  {
       connectSigs(this, impl_);
  } 
  
  EventProcessor::EventProcessor(int argc, char* argv[], const string& config):
    impl_(new FwkImpl(argc,argv,config))
  {
       connectSigs(this, impl_);
  }

  EventProcessor::EventProcessor(int argc, char* argv[],
                                 const ServiceToken& iToken,serviceregistry::ServiceLegacy iLegacy):
     impl_(new FwkImpl(argc,argv,iToken,iLegacy))
  {
        connectSigs(this, impl_);
  } 
  
  EventProcessor::EventProcessor(const string& config,
                                 const ServiceToken& iToken,serviceregistry::ServiceLegacy iLegacy):
     impl_(new FwkImpl(config,iToken,iLegacy))
  {
        connectSigs(this, impl_);
  } 
  
  EventProcessor::EventProcessor(int argc, char* argv[], const string& config,
                                 const ServiceToken& iToken,serviceregistry::ServiceLegacy iLegacy):
     impl_(new FwkImpl(argc,argv,config,iToken,iLegacy))
  {
        connectSigs(this, impl_);
  }
  
  EventProcessor::~EventProcessor()
  {
    //make the service's available while everything is being deleted
    ServiceToken token = impl_->serviceToken_;
    ServiceRegistry::Operate op(token); 
    delete impl_;
  }

  EventProcessor::StatusCode
  EventProcessor::run(unsigned long numberToProcess)
  {
    return impl_->run(numberToProcess);
  }
  
  void
  EventProcessor::beginJob() 
  {
    impl_->beginJob();
  }

  bool
  EventProcessor::endJob() 
  {
    return impl_->endJob();
  }

  InputSource&
  EventProcessor::getInputSource()
  {
    return *impl_->input_;
  }
}
