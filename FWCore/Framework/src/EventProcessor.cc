#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>

#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"

#include "PluginManager/PluginManager.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/ScheduleBuilder.h"
#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/Schedule.h"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/Log.h"


using namespace std;
using boost::shared_ptr;
using edm::serviceregistry::ServiceLegacy; 
using edm::serviceregistry::kOverlapIsError;

namespace edm {

  typedef vector<string>   StrVec;
  typedef list<string>     StrList;
  typedef Worker*          WorkerPtr;
  typedef list<WorkerPtr>  WorkerList;
  typedef list<WorkerList> PathList;

  struct CommonParams
  {
    CommonParams():version_(),pass_() { }
    CommonParams(const string& name,unsigned long ver,unsigned long pass):
      processName_(name),version_(ver),pass_(pass) { }

    string                  processName_;
    unsigned long           version_;
    unsigned long           pass_;
  }; // struct CommonParams


  // temporary function because we do not know how to do this
  unsigned long getVersion() { return 0; }

  shared_ptr<InputSource> 
  makeInput(ParameterSet const& params,
	    const CommonParams& common,
	    ProductRegistry& preg)
  {
    // find single source
    bool sourceSpecified = false;
    try {
      const std::string& processName = params.getUntrackedParameter<string>("@process_name");
      ParameterSet main_input = 
	params.getParameter<ParameterSet>("@main_input");

      // Fill in "ModuleDescription", in case the input source produces any EDproducts,
      // which would be registered in the ProductRegistry.
      ModuleDescription md;
      md.pid = main_input.id();
      md.moduleName_ = main_input.template getUntrackedParameter<std::string>("@module_type");
      // There is no module label for the unnamed input source, so just use the module name.
      md.moduleLabel_ = md.moduleName_;
      md.processName_ = processName;
//#warning version and pass are hardcoded
      md.versionNumber_ = 1;
      md.pass = 1; 

      sourceSpecified = true;
      InputSourceDescription isdesc(common.processName_,common.pass_,preg);
      shared_ptr<InputSource> input
	(InputSourceFactory::get()->makeInputSource(main_input, isdesc).release());
      input->addToRegistry(md);
    
      return input;
    } catch(const edm::Exception& iException) {
      if(sourceSpecified == false && errors::Configuration == iException.categoryCode()) {
        throw edm::Exception(errors::Configuration, "NoInputSource")
	  <<"No main input source found in configuration.  Please add an input source via 'source = ...' in the configuration file.\n";
      } else {
        throw;
      }
    }
    return shared_ptr<InputSource>();
  }
  
  static
  std::auto_ptr<eventsetup::EventSetupProvider>
   makeEventSetupProvider(ParameterSet const& params)
  {
     using namespace std;
     using namespace edm::eventsetup;
     vector<string> prefers = params.getParameter<vector<string> >("@all_esprefers");
     if(prefers.empty()){
        return std::auto_ptr<EventSetupProvider>(new EventSetupProvider());
     }

     EventSetupProvider::PreferredProviderInfo preferInfo;
     EventSetupProvider::RecordToDataMap recordToData;
     //recordToData.insert(std::make_pair(std::string("DummyRecord"),
     //                                   std::make_pair(std::string("DummyData"),std::string())));
     //preferInfo[ComponentDescription("DummyProxyProvider","",false)]=recordToData;

     for(vector<string>::iterator itName = prefers.begin();
         itName != prefers.end();
         ++itName) 
     {
        recordToData.clear();
	ParameterSet preferPSet = params.getParameter<ParameterSet>(*itName);
        std::vector<std::string> recordNames = preferPSet.getParameterNames();
        for(std::vector<std::string>::iterator itRecordName = recordNames.begin();
            itRecordName != recordNames.end();
            ++itRecordName) {
           if( (*itRecordName)[0]=='@'){
              //this is a 'hidden parameter' so skip it
              continue;
           }
           //this should be a record name with its info
           try {
              std::vector<std::string> dataInfo = preferPSet.getParameter<vector<std::string> >(*itRecordName);

              if(dataInfo.empty()) {
                 //FUTURE: empty should just mean all data
                 throw cms::Exception("Configuration")<<"The record named "<<*itRecordName<<" specifies no data items";
              }
              //FUTURE: 'any' should be a special name
              for(std::vector<std::string>::iterator itDatum = dataInfo.begin();
                  itDatum != dataInfo.end();
                  ++itDatum){
                 std::string datumName(*itDatum, 0, itDatum->find_first_of("/"));
                 std::string labelName;
                 if(itDatum->size() != datumName.size()) {
                    labelName = std::string(*itDatum, datumName.size()+1);
                 }
                 recordToData.insert(std::make_pair(std::string(*itRecordName),
                                                    std::make_pair(datumName,
                                                                   labelName)));
              }
           } catch(const cms::Exception& iException) {
              cms::Exception theError("ESPreferConfigurationError");
              theError<<"While parsing the es_prefer statement for type="<<preferPSet.getParameter<std::string>("@module_type")
                 <<" label=\""<<preferPSet.getParameter<std::string>("@module_label")<<"\" an error occurred.";
              theError.append(iException);
              throw theError;
           }
        }
        preferInfo[ComponentDescription(preferPSet.getParameter<std::string>("@module_type"),
                                        preferPSet.getParameter<std::string>("@module_label"),
                                        false)]
           =recordToData;
     }
     return std::auto_ptr<EventSetupProvider>(new EventSetupProvider(&preferInfo));
  }
  
  void 
  fillEventSetupProvider(edm::eventsetup::EventSetupProvider& cp,
			 ParameterSet const& params,
			 const CommonParams& common)
  {
    using namespace std;
    using namespace edm::eventsetup;
    vector<string> providers = params.getParameter<vector<string> >("@all_esmodules");
    for(vector<string>::iterator itName = providers.begin();
	itName != providers.end();
	++itName) 
      {
	ParameterSet providerPSet = params.getParameter<ParameterSet>(*itName);
	ModuleFactory::get()->addTo(cp, 
				    providerPSet, 
				    common.processName_, 
				    common.version_, 
				    common.pass_);
      }
    
    vector<string> sources = params.getParameter<vector<string> >("@all_essources");
    for(vector<string>::iterator itName = sources.begin();
	itName != sources.end();
	++itName) 
      {
	ParameterSet providerPSet = params.getParameter<ParameterSet>(*itName);
	SourceFactory::get()->addTo(cp, 
				    providerPSet, 
				    common.processName_, 
				    common.version_, 
				    common.pass_);
    }
  }


  //need a wrapper to let me 'copy' references to EventSetup
  namespace eventprocessor 
  {
    struct ESRefWrapper 
    {
      EventSetup const & es_;
      ESRefWrapper(EventSetup const &iES) : es_(iES) {}
      operator const EventSetup&() { return es_; }
    };
  }

  using eventprocessor::ESRefWrapper;

  //----------------------------------------------------------------------
  // Implementation of FwkImpl, the 'pimpl' for EventProcessor
  //----------------------------------------------------------------------
  //
  // right now we only support a pset string from constructor or
  // pset read from file

  struct DoPluginInit
  {
	DoPluginInit()
	{ seal::PluginManager::get()->initialise();
	  // std::cerr << "Initialized pligin manager" << std::endl;
	}
  };

  class FwkImpl
  {
  public:
    explicit FwkImpl(const string& config,
                     const ServiceToken& = ServiceToken(),
                     ServiceLegacy=kOverlapIsError);

    EventProcessor::StatusCode run(unsigned long numberToProcess);
    void                       beginJob();
    bool                       endJob();


    ServiceToken   getToken();
    void           connectSigs(EventProcessor* ep);
    InputSource&   getInputSource();

  private:

    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.    
    //shared_ptr<ParameterSet>        params_;
	DoPluginInit  plug_init_;
    CommonParams                    common_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    WorkerRegistry wreg_;
    SignallingProductRegistry       preg_;

    ServiceToken                    serviceToken_;
    shared_ptr<InputSource>         input_;
    std::auto_ptr<Schedule> sched_;
    std::auto_ptr<eventsetup::EventSetupProvider>  
                                    esp_;    

    bool                            emittedBeginJob_;
    ActionTable                     act_table_;
  }; // class FwkImpl

  // ---------------------------------------------------------------

  FwkImpl::FwkImpl(const string& config,
                   const ServiceToken& iToken, 
		   ServiceLegacy iLegacy) :
    //params_(),
    common_(),
    actReg_(new ActivityRegistry),
    wreg_(actReg_),
    preg_(),
    serviceToken_(),
    input_(),
    sched_(),
    esp_(),
    emittedBeginJob_(false),
    act_table_()
  {
    // TODO: Fix const-correctness. The ParameterSets that are
    // returned here should be const, so that we can be sure they are
    // not modified.

    shared_ptr<vector<ParameterSet> > pServiceSets;
    shared_ptr<ParameterSet>          params_; // change this name!
    makeParameterSets(config, params_, pServiceSets);

    //create the services
    serviceToken_ = ServiceRegistry::createSet(*pServiceSets,iToken,iLegacy);
    serviceToken_.connectTo(*actReg_);
     
    //add the ProductRegistry as a service ONLY for the construction phase
    typedef serviceregistry::ServiceWrapper<ConstProductRegistry> w_CPR;
    shared_ptr<w_CPR>
      reg(new w_CPR( std::auto_ptr<ConstProductRegistry>(new ConstProductRegistry(preg_))));
    ServiceToken tempToken( ServiceRegistry::createContaining(reg, 
							      serviceToken_, 
							      kOverlapIsError));

    // the next thing is ugly: pull out the trigger path pset and 
    // create a service and extra token for it
    string proc_name = params_->getParameter<string>("@process_name");

    typedef edm::service::TriggerNamesService TNS;
    typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

    ParameterSet trigger_paths =
      (*params_).getUntrackedParameter<ParameterSet>("@trigger_paths");
    shared_ptr<w_TNS> tnsptr
      (new w_TNS( std::auto_ptr<TNS>(new TNS(trigger_paths,proc_name))));
    ServiceToken tempToken2(ServiceRegistry::createContaining(tnsptr, 
							      tempToken, 
							      kOverlapIsError));
    //make the services available
    ServiceRegistry::Operate operate(tempToken2);
     
    //params_ = builder.getProcessPSet();
    act_table_ = ActionTable(*params_);
    common_ = CommonParams(proc_name,
			   getVersion(), // this is not written for real yet
			   0); // Where does it come from?
     
    input_= makeInput(*params_, common_, preg_);     
    sched_ = std::auto_ptr<Schedule>(new Schedule(*params_,wreg_,
						  preg_,act_table_,
						  actReg_));

    esp_ = makeEventSetupProvider(*params_);
    fillEventSetupProvider(*esp_, *params_, common_);
    //   initialize(iToken,iLegacy);
    FDEBUG(2) << params_->toString() << std::endl;
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
	IOVSyncValue ts(pep->id(), pep->time());
	EventSetup const& es = esp_->eventSetupForInstance(ts);

	sched_->runOneEvent(*pep.get(),es);
      }

    return 0;
  }

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
      EventSetup const& es = esp_->eventSetupForInstance(IOVSyncValue::beginOfTime());
      sched_->beginJob(es);
      emittedBeginJob_ = true;
      actReg_->postBeginJobSignal_();
    }
  }

  bool
  FwkImpl::endJob() 
  {
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);  
    bool returnValue = false;

    try
      {
	sched_->endJob();
	returnValue=true;
      }
    catch(cms::Exception& iException)
      {
	LogError(iException.category())
	  << "Caught cms::Exception in endJob: "<< iException.what() << "\n";
      }
    catch(std::exception& iException)
      {
	LogError("std::exception")
	  << "Caught std::exception in endJob: "<< iException.what() << "\n";
      }
    catch(...)
      {
	LogError("ignored_exception")
	  << "Caught unknown exception in endJob. (ignoring it!!!!)\n";
      }
    
    actReg_->postEndJobSignal_();
    return returnValue;
  }

  ServiceToken
  FwkImpl::getToken()
  {
    return serviceToken_;
  }

  void
  FwkImpl::connectSigs(EventProcessor* ep)
  {
    // When the FwkImpl signals are given, pass them to the
    // appropriate EventProcessor signals so that the outside world
    // can see the signal.
    actReg_->preProcessEventSignal_.connect(ep->preProcessEventSignal);
    actReg_->postProcessEventSignal_.connect(ep->postProcessEventSignal);
  }

  InputSource&
  FwkImpl::getInputSource()
  {
    return *input_;
  }

  //----------------------------------------------------------------------
  // Implementation of EventProcessor
  //----------------------------------------------------------------------
  EventProcessor::EventProcessor(const string& config) :
    impl_(new FwkImpl(config, 
		      ServiceToken(), //  no pre-made services
		      kOverlapIsError))
  {
    impl_->connectSigs(this);
  } 
  
  EventProcessor::EventProcessor(const string& config,
				 const ServiceToken& iToken,
				 ServiceLegacy iLegacy):
    impl_(new FwkImpl(config,iToken,iLegacy))
  {
    impl_->connectSigs(this);
  } 
  
  EventProcessor::~EventProcessor()
  {
    // Make the services available while everything is being deleted.
    ServiceToken token = impl_->getToken();
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
    return impl_->getInputSource();
  }

}
