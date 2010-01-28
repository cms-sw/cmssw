
#include "FWCore/Framework/interface/EventProcessor.h"

#include <exception>
#include <utility>
#include <iostream>
#include <iomanip>

//Used for forking
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "boost/bind.hpp"
#include "boost/thread/xtime.hpp"

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/Framework/src/Breakpoints.h"
#include "FWCore/Framework/src/InputSourceFactory.h"

#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/EDLooper.h"

#include "FWCore/Framework/src/EPStates.h"

#include "FWCore/Framework/interface/EventSetupRecord.h"

#include "BeginJobCleanup.h"

using edm::serviceregistry::ServiceLegacy; 
using edm::serviceregistry::kOverlapIsError;

namespace edm {

  namespace event_processor {

    class StateSentry {
    public:
      StateSentry(EventProcessor* ep) : ep_(ep), success_(false) { }
      ~StateSentry() {if(!success_) ep_->changeState(mException);}
      void succeeded() {success_ = true;}

    private:
      EventProcessor* ep_;
      bool success_;
    };
  }

  using namespace event_processor;
  using namespace edm::service;

  namespace {

    // the next two tables must be kept in sync with the state and
    // message enums from the header

    char const* stateNames[] = {
      "Init",
      "JobReady",
      "RunGiven",
      "Running",
      "Stopping",
      "ShuttingDown",
      "Done",
      "JobEnded",
      "Error",
      "ErrorEnded",
      "End",
      "Invalid"
    };

    char const* msgNames[] = {
      "SetRun",
      "Skip",
      "RunAsync",
      "Run(ID)",
      "Run(count)",
      "BeginJob",
      "StopAsync",
      "ShutdownAsync",
      "EndJob",
      "CountComplete",
      "InputExhausted",
      "StopSignal",
      "ShutdownSignal",
      "Finished",
      "Any",
      "dtor",
      "Exception",
      "Rewind"
    };
  }
    // IMPORTANT NOTE:
    // the mAny messages are special, they must appear last in the
    // table if multiple entries for a CurrentState are present.
    // the changeState function does not use the mAny yet!!!

    struct TransEntry {
      State current;
      Msg   message;
      State final;
    };

    // we should use this information to initialize a two dimensional
    // table of t[CurrentState][Message] = FinalState

    /*
      the way this is current written, the async run can thread function
      can return in the "JobReady" state - but not yet cleaned up.  The
      problem is that only when stop/shutdown async is called is the 
      thread cleaned up. But the stop/shudown async functions attempt
      first to change the state using messages that are not valid in
      "JobReady" state.

      I think most of the problems can be solved by using two states
      for "running": RunningS and RunningA (sync/async). The problems
      seems to be the all the transitions out of running for both
      modes of operation.  The other solution might be to only go to
      "Stopping" from Running, and use the return code from "run_p" to
      set the final state.  If this is used, then in the sync mode the
      "Stopping" state would be momentary.

     */

    TransEntry table[] = {
    // CurrentState   Message         FinalState
    // -----------------------------------------
      { sInit,          mException,      sError },
      { sInit,          mBeginJob,       sJobReady },
      { sJobReady,      mException,      sError },
      { sJobReady,      mSetRun,         sRunGiven },
      { sJobReady,      mInputRewind,    sRunning },
      { sJobReady,      mSkip,           sRunning },
      { sJobReady,      mRunID,          sRunning },
      { sJobReady,      mRunCount,       sRunning },
      { sJobReady,      mEndJob,         sJobEnded },
      { sJobReady,      mBeginJob,       sJobReady },
      { sJobReady,      mDtor,           sEnd },    // should this be allowed?

      { sJobReady,      mStopAsync,      sJobReady },
      { sJobReady,      mCountComplete,  sJobReady },
      { sJobReady,      mFinished,       sJobReady },

      { sRunGiven,      mException,      sError },
      { sRunGiven,      mRunAsync,       sRunning },
      { sRunGiven,      mBeginJob,       sRunGiven },
      { sRunGiven,      mShutdownAsync,  sShuttingDown },
      { sRunGiven,      mStopAsync,      sStopping },
      { sRunning,       mException,      sError },
      { sRunning,       mStopAsync,      sStopping },
      { sRunning,       mShutdownAsync,  sShuttingDown },
      { sRunning,       mShutdownSignal, sShuttingDown },
      { sRunning,       mCountComplete,  sStopping }, // sJobReady 
      { sRunning,       mInputExhausted, sStopping }, // sJobReady

      { sStopping,      mInputRewind,    sRunning }, // The looper needs this
      { sStopping,      mException,      sError },
      { sStopping,      mFinished,       sJobReady },
      { sStopping,      mCountComplete,  sJobReady },
      { sStopping,      mShutdownSignal, sShuttingDown },
      { sStopping,      mStopAsync,      sStopping },     // stay
      { sStopping,      mInputExhausted, sStopping },     // stay
      //{ sStopping,      mAny,            sJobReady },     // <- ??????
      { sShuttingDown,  mException,      sError },
      { sShuttingDown,  mShutdownSignal, sShuttingDown },
      { sShuttingDown,  mCountComplete,  sDone }, // needed?
      { sShuttingDown,  mInputExhausted, sDone }, // needed?
      { sShuttingDown,  mFinished,       sDone },
      //{ sShuttingDown,  mShutdownAsync,  sShuttingDown }, // only one at
      //{ sShuttingDown,  mStopAsync,      sShuttingDown }, // a time
      //{ sShuttingDown,  mAny,            sDone },         // <- ??????
      { sDone,          mEndJob,         sJobEnded },
      { sDone,          mException,      sError },
      { sJobEnded,      mDtor,           sEnd },
      { sJobEnded,      mException,      sError },
      { sError,         mEndJob,         sError },   // funny one here
      { sError,         mDtor,           sError },   // funny one here
      { sInit,          mDtor,           sEnd },     // for StorM dummy EP
      { sStopping,      mShutdownAsync,  sShuttingDown }, // For FUEP tests
      { sInvalid,       mAny,            sInvalid }
    };


    // Note: many of the messages generate the mBeginJob message first 
    //  mRunID, mRunCount, mSetRun

  // ---------------------------------------------------------------
  boost::shared_ptr<InputSource> 
  makeInput(ParameterSet& params,
	    EventProcessor::CommonParams const& common,
	    ProductRegistry& preg,
	    PrincipalCache& pCache,
            boost::shared_ptr<ActivityRegistry> areg,
	    boost::shared_ptr<ProcessConfiguration> processConfiguration) {
    ParameterSet * main_input = params.getPSetForUpdate("@main_input");
    if (main_input == 0) {
      throw edm::Exception(errors::Configuration, "FailedInputSource")
	<< "Configuration of main input source has failed\n";
    }

    std::string modtype;
    try {
      modtype = main_input->getParameter<std::string>("@module_type");
      std::auto_ptr<edm::ParameterSetDescriptionFillerBase> filler(
        edm::ParameterSetDescriptionFillerPluginFactory::get()->create(modtype));
      ConfigurationDescriptions descriptions(filler->baseType());
      filler->fill(descriptions);
      descriptions.validate(*main_input, std::string("source"));
    }
    catch (cms::Exception& iException) {
      edm::Exception toThrow(errors::Configuration, "Failed validating main input source configuration.");
      toThrow << "\nSource plugin name is \"" << modtype << "\"\n";
      toThrow.append(iException);
      throw toThrow;
    }

    main_input->registerIt();
 
    // Fill in "ModuleDescription", in case the input source produces
    // any EDproducts, which would be registered in the ProductRegistry.
    // Also fill in the process history item for this process.
    // There is no module label for the unnamed input source, so 
    // just use "source".
    // Only the tracked parameters belong in the process configuration.
    ModuleDescription md(main_input->id(),
                         main_input->getParameter<std::string>("@module_type"),
		         "source",
		         processConfiguration);

    InputSourceDescription isdesc(md, preg, pCache, areg, common.maxEventsInput_, common.maxLumisInput_);
    areg->preSourceConstructionSignal_(md);
    boost::shared_ptr<InputSource> input(InputSourceFactory::get()->makeInputSource(*main_input, isdesc).release());
    areg->postSourceConstructionSignal_(md);
      
    return input;
  }
  
  // ---------------------------------------------------------------
  static
  std::auto_ptr<eventsetup::EventSetupProvider>
  makeEventSetupProvider(ParameterSet const& params) {
    using namespace edm::eventsetup;
    std::vector<std::string> prefers =
      params.getParameter<std::vector<std::string> >("@all_esprefers");

    if(prefers.empty()) {
      return std::auto_ptr<EventSetupProvider>(new EventSetupProvider());
    }

    EventSetupProvider::PreferredProviderInfo preferInfo;
    EventSetupProvider::RecordToDataMap recordToData;

    //recordToData.insert(std::make_pair(std::string("DummyRecord"),
    //      std::make_pair(std::string("DummyData"), std::string())));
    //preferInfo[ComponentDescription("DummyProxyProvider", "", false)]=
    //      recordToData;

    for(std::vector<std::string>::iterator itName = prefers.begin(), itNameEnd = prefers.end();
	itName != itNameEnd;
	++itName) {
        recordToData.clear();
	ParameterSet preferPSet = params.getParameter<ParameterSet>(*itName);
        std::vector<std::string> recordNames = preferPSet.getParameterNames();
        for(std::vector<std::string>::iterator itRecordName = recordNames.begin(),
	    itRecordNameEnd = recordNames.end();
            itRecordName != itRecordNameEnd;
            ++itRecordName) {

	  if((*itRecordName)[0] == '@') {
	    //this is a 'hidden parameter' so skip it
	    continue;
	  }

	  //this should be a record name with its info
	  try {
	    std::vector<std::string> dataInfo =
	      preferPSet.getParameter<std::vector<std::string> >(*itRecordName);
	    
	    if(dataInfo.empty()) {
	      //FUTURE: empty should just mean all data
	      throw edm::Exception(errors::Configuration)
		<< "The record named "
		<< *itRecordName << " specifies no data items";
	    }
	    //FUTURE: 'any' should be a special name
	    for(std::vector<std::string>::iterator itDatum = dataInfo.begin(),
	        itDatumEnd = dataInfo.end();
		itDatum != itDatumEnd;
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
	  } catch(cms::Exception const& iException) {
	    cms::Exception theError("ESPreferConfigurationError");
	    theError << "While parsing the es_prefer statement for type="
		     << preferPSet.getParameter<std::string>("@module_type")
		     << " label=\""
		     << preferPSet.getParameter<std::string>("@module_label")
		     << "\" an error occurred.";
	    theError.append(iException);
	    throw theError;
	  }
        }
        preferInfo[ComponentDescription(preferPSet.getParameter<std::string>("@module_type"),
                                        preferPSet.getParameter<std::string>("@module_label"),
                                        false)]
	  = recordToData;
      }
    return std::auto_ptr<EventSetupProvider>(new EventSetupProvider(&preferInfo));
  }
  
  // ---------------------------------------------------------------
  void 
  fillEventSetupProvider(eventsetup::EventSetupProvider& cp,
			 ParameterSet& params,
			 EventProcessor::CommonParams const& common) {
    using namespace edm::eventsetup;
    std::vector<std::string> providers =
      params.getParameter<std::vector<std::string> >("@all_esmodules");

    for(std::vector<std::string>::iterator itName = providers.begin(), itNameEnd = providers.end();
	itName != itNameEnd;
	++itName) {
      ParameterSet * providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      ModuleFactory::get()->addTo(cp, 
				    *providerPSet, 
				    common.processName_, 
				    common.releaseVersion_, 
				    common.passID_);
      }
    
    std::vector<std::string> sources = 
      params.getParameter<std::vector<std::string> >("@all_essources");

    for(std::vector<std::string>::iterator itName = sources.begin(), itNameEnd = sources.end();
	itName != itNameEnd;
	++itName) {
      ParameterSet * providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      SourceFactory::get()->addTo(cp, 
				    *providerPSet, 
				    common.processName_, 
				    common.releaseVersion_, 
				    common.passID_);
    }
  }

  // ---------------------------------------------------------------
  boost::shared_ptr<EDLooper> 
  fillLooper(eventsetup::EventSetupProvider& cp,
			 ParameterSet& params,
			 EventProcessor::CommonParams const& common) {
    using namespace edm::eventsetup;
    boost::shared_ptr<EDLooper> vLooper;
    
    std::vector<std::string> loopers =
      params.getParameter<std::vector<std::string> >("@all_loopers");

    if(loopers.size() == 0) {
       return vLooper;
    }
   
    assert(1 == loopers.size());

    for(std::vector<std::string>::iterator itName = loopers.begin(), itNameEnd = loopers.end();
	itName != itNameEnd;
	++itName) {

      ParameterSet * providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      vLooper = LooperFactory::get()->addTo(cp, 
				    *providerPSet, 
				    common.processName_, 
				    common.releaseVersion_, 
				    common.passID_);
      }
      return vLooper;
    
  }

  // ---------------------------------------------------------------
  EventProcessor::EventProcessor(std::string const& config,
				ServiceToken const& iToken, 
				serviceregistry::ServiceLegacy iLegacy,
			        std::vector<std::string> const& defaultServices,
				std::vector<std::string> const& forcedServices) :
    preProcessEventSignal_(),
    postProcessEventSignal_(),
    maxEventsPset_(),
    maxLumisPset_(),
    actReg_(new ActivityRegistry),
    preg_(new SignallingProductRegistry),
    serviceToken_(),
    input_(),
    esp_(),
    act_table_(),
    wreg_(actReg_),
    processConfiguration_(),
    schedule_(),
    state_(sInit),
    event_loop_(),
    state_lock_(),
    stop_lock_(),
    stopper_(),
    stop_count_(-1),
    last_rc_(epSuccess),
    last_error_text_(),
    id_set_(false),
    event_loop_id_(),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    shouldWeStop_(false),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1) {
    boost::shared_ptr<ProcessDesc> processDesc = PythonProcessDesc(config).processDesc();
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, iToken, iLegacy);
  }

  EventProcessor::EventProcessor(std::string const& config,
			        std::vector<std::string> const& defaultServices,
				std::vector<std::string> const& forcedServices) :
    preProcessEventSignal_(),
    postProcessEventSignal_(),
    maxEventsPset_(),
    maxLumisPset_(),
    actReg_(new ActivityRegistry),
    preg_(new SignallingProductRegistry),
    serviceToken_(),
    input_(),
    esp_(),
    act_table_(),
    wreg_(actReg_),
    processConfiguration_(),
    schedule_(),
    state_(sInit),
    event_loop_(),
    state_lock_(),
    stop_lock_(),
    stopper_(),
    stop_count_(-1),
    last_rc_(epSuccess),
    last_error_text_(),
    id_set_(false),
    event_loop_id_(),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    shouldWeStop_(false),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1) {
    boost::shared_ptr<ProcessDesc> processDesc = PythonProcessDesc(config).processDesc();
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
  }

  EventProcessor::EventProcessor(boost::shared_ptr<ProcessDesc>& processDesc,
                 ServiceToken const& token,
                 serviceregistry::ServiceLegacy legacy) :
    preProcessEventSignal_(),
    postProcessEventSignal_(),
    maxEventsPset_(),
    maxLumisPset_(),
    actReg_(new ActivityRegistry),
    preg_(new SignallingProductRegistry),
    serviceToken_(),
    input_(),
    esp_(),
    act_table_(),
    wreg_(actReg_),
    processConfiguration_(),
    schedule_(),
    state_(sInit),
    event_loop_(),
    state_lock_(),
    stop_lock_(),
    stopper_(),
    stop_count_(-1),
    last_rc_(epSuccess),
    last_error_text_(),
    id_set_(false),
    event_loop_id_(),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    shouldWeStop_(false),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false) {
    init(processDesc, token, legacy);
  }


  EventProcessor::EventProcessor(std::string const& config, bool isPython):
    preProcessEventSignal_(),
    postProcessEventSignal_(),
    maxEventsPset_(),
    maxLumisPset_(),
    actReg_(new ActivityRegistry),
    preg_(new SignallingProductRegistry),
    serviceToken_(),
    input_(),
    esp_(),
    act_table_(),
    wreg_(actReg_),
    processConfiguration_(),
    schedule_(),
    state_(sInit),
    event_loop_(),
    state_lock_(),
    stop_lock_(),
    stopper_(),
    stop_count_(-1),
    last_rc_(epSuccess),
    last_error_text_(),
    id_set_(false),
    event_loop_id_(),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    shouldWeStop_(false),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false) {
    if(isPython) {
      boost::shared_ptr<ProcessDesc> processDesc = PythonProcessDesc(config).processDesc();
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
    else {
      boost::shared_ptr<ProcessDesc> processDesc(new ProcessDesc(config));
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
  }

  void
  EventProcessor::init(boost::shared_ptr<ProcessDesc>& processDesc,
			ServiceToken const& iToken, 
			serviceregistry::ServiceLegacy iLegacy) {

    // The BranchIDListRegistry and ProductIDListRegistry are indexed registries, and are singletons.
    //  They must be cleared here because some processes run multiple EventProcessors in succession.
    BranchIDListHelper::clearRegistries();

    boost::shared_ptr<ParameterSet> parameterSet = processDesc->getProcessPSet();
    

    ParameterSet optionsPset(parameterSet->getUntrackedParameter<ParameterSet>("options", ParameterSet()));
    fileMode_ = optionsPset.getUntrackedParameter<std::string>("fileMode", "");
    handleEmptyRuns_ = optionsPset.getUntrackedParameter<bool>("handleEmptyRuns", true);
    handleEmptyLumis_ = optionsPset.getUntrackedParameter<bool>("handleEmptyLumis", true);
    ParameterSet forking = optionsPset.getUntrackedParameter<ParameterSet>("multiProcesses", ParameterSet());
    numberOfForkedChildren_ = forking.getUntrackedParameter<int>("maxChildProcesses", 0);
    numberOfSequentialEventsPerChild_ = forking.getUntrackedParameter<unsigned int>("maxSequentialEventsPerChild", 1);
    std::vector<ParameterSet> excluded = forking.getUntrackedParameter<std::vector<ParameterSet> >("eventSetupDataToExcludeFromPrefetching",std::vector<ParameterSet>());
    for(std::vector<ParameterSet>::const_iterator itPS=excluded.begin(),itPSEnd=excluded.end();
        itPS != itPSEnd;
        ++itPS) {
      eventSetupDataToExcludeFromPrefetching_[itPS->getUntrackedParameter<std::string>("record")].insert(
                                                std::make_pair(itPS->getUntrackedParameter<std::string>("type","*"),
                                                               itPS->getUntrackedParameter<std::string>("label","")));
    }
    
    maxEventsPset_ = parameterSet->getUntrackedParameter<ParameterSet>("maxEvents", ParameterSet());
    maxLumisPset_ = parameterSet->getUntrackedParameter<ParameterSet>("maxLuminosityBlocks", ParameterSet());

    boost::shared_ptr<std::vector<ParameterSet> > pServiceSets = processDesc->getServicesPSets();
    //makeParameterSets(config, parameterSet, pServiceSets);

    //create the services
    ServiceToken tempToken(ServiceRegistry::createSet(*pServiceSets, iToken, iLegacy));

    // Copy slots that hold all the registered callback functions like
    // PostBeginJob into an ActivityRegistry that is owned by EventProcessor
    tempToken.copySlotsTo(*actReg_); 
    
    //add the ProductRegistry as a service ONLY for the construction phase
    typedef serviceregistry::ServiceWrapper<ConstProductRegistry> w_CPR;
    boost::shared_ptr<w_CPR>
      reg(new w_CPR(std::auto_ptr<ConstProductRegistry>(new ConstProductRegistry(*preg_))));
    ServiceToken tempToken2(ServiceRegistry::createContaining(reg, 
							      tempToken, 
							      kOverlapIsError));

    // the next thing is ugly: pull out the trigger path pset and 
    // create a service and extra token for it
    std::string processName = parameterSet->getParameter<std::string>("@process_name");

    typedef service::TriggerNamesService TNS;
    typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

    boost::shared_ptr<w_TNS> tnsptr
      (new w_TNS(std::auto_ptr<TNS>(new TNS(*parameterSet))));

    serviceToken_ = ServiceRegistry::createContaining(tnsptr, 
						    tempToken2, 
						    kOverlapIsError);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);
     
    //parameterSet = builder.getProcessPSet();
    act_table_ = ActionTable(*parameterSet);
    CommonParams common = CommonParams(processName,
			   getReleaseVersion(),
			   getPassID(),
    			   maxEventsPset_.getUntrackedParameter<int>("input", -1),
    			   maxLumisPset_.getUntrackedParameter<int>("input", -1));

    esp_ = makeEventSetupProvider(*parameterSet);
    fillEventSetupProvider(*esp_, *parameterSet, common);

    looper_ = fillLooper(*esp_, *parameterSet, common);
    if (looper_) looper_->setActionTable(&act_table_);
    
    processConfiguration_.reset(new ProcessConfiguration(processName, getReleaseVersion(), getPassID()));
    input_ = makeInput(*parameterSet, common, *preg_, principalCache_, actReg_, processConfiguration_);
    schedule_ = std::auto_ptr<Schedule>
      (new Schedule(parameterSet,
		    ServiceRegistry::instance().get<TNS>(),
		    wreg_,
		    *preg_,
		    act_table_,
		    actReg_,
		    processConfiguration_));

    //   initialize(iToken, iLegacy);
    FDEBUG(2) << parameterSet << std::endl;
    connectSigs(this);
    ProcessConfigurationRegistry::instance()->insertMapped(*processConfiguration_);
    BranchIDListHelper::updateRegistries(*preg_);
  }

  EventProcessor::~EventProcessor() {
    // Make the services available while everything is being deleted.
    ServiceToken token = getToken();
    ServiceRegistry::Operate op(token); 

    // The state machine should have already been cleaned up
    // and destroyed at this point by a call to EndJob or
    // earlier when it completed processing events, but if it
    // has not been we'll take care of it here at the last moment.
    // This could cause problems if we are already handling an
    // exception and another one is thrown here ...  For a critical
    // executable the solution to this problem is for the code using
    // the EventProcessor to explicitly call EndJob or use runToCompletion,
    // then the next line of code is never executed.
    terminateMachine();

    try {
      changeState(mDtor);
    }
    catch(cms::Exception& e) {
      LogError("System")
	<< e.explainSelf() << "\n";
    }

    // manually destroy all these thing that may need the services around
    esp_.reset();
    schedule_.reset();
    input_.reset();
    looper_.reset();
    wreg_.clear();
    actReg_.reset();

    pset::Registry* psetRegistry = pset::Registry::instance();
    psetRegistry->data().clear();
    psetRegistry->extra().setID(ParameterSetID());

    EntryDescriptionRegistry::instance()->data().clear();
    ParentageRegistry::instance()->data().clear();
    ProcessConfigurationRegistry::instance()->data().clear();
    ProcessHistoryRegistry::instance()->data().clear();
    BranchIDListHelper::clearRegistries();
  }

  void
  EventProcessor::rewind() {
    beginJob(); //make sure this was called
    changeState(mStopAsync);
    changeState(mInputRewind);
    {
      StateSentry toerror(this);

      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      
      {
	input_->repeat();
        input_->rewind();
      }
      changeState(mCountComplete);
      toerror.succeeded();
    }
    changeState(mFinished);
  }

  bool
  EventProcessor::doOneEvent(EventID const& id) {
    boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_, *processConfiguration_));
    principalCache_.insert(ep);
    EventPrincipal* pep = 0;
    try {
      pep = input_->readEvent(id);
    }
    catch(cms::Exception& e) {
      actions::ActionCodes action = act_table_.find(e.rootCause());
      if (action == actions::Rethrow) {
 	throw;
      } else {
        LogWarning(e.category())
          << "an exception occurred and all paths for the event are being skipped: \n"
          << e.what();
        return true;
      }
    }
    if (pep != 0) {
      IOVSyncValue ts(ep->id(), ep->time());
      EventSetup const& es = esp_->eventSetupForInstance(ts);
      schedule_->processOneOccurrence<OccurrenceTraits<EventPrincipal, BranchActionBegin> >(*ep, es);
    }
    return (pep != 0);
  }

  EventProcessor::StatusCode
  EventProcessor::run(int numberEventsToProcess, bool) {
    return runEventCount(numberEventsToProcess);
  }
  
  EventProcessor::StatusCode
  EventProcessor::run(EventID const& id) {
    beginJob(); //make sure this was called
    changeState(mRunID);
    StateSentry toerror(this);
    Status rc = epSuccess;

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    if(!doOneEvent(id)) {
      changeState(mInputExhausted);
    } else {
      changeState(mCountComplete);
      rc = epInputComplete;
    }
    toerror.succeeded();
    changeState(mFinished);
    return rc;
  }

  EventProcessor::StatusCode
  EventProcessor::skip(int numberToSkip) {
    beginJob(); //make sure this was called
    changeState(mSkip);
    {
      StateSentry toerror(this);

      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      
      {
        input_->skipEvents(numberToSkip);
      }
      changeState(mCountComplete);
      toerror.succeeded();
    }
    changeState(mFinished);
    return epSuccess;
  }

  void
  EventProcessor::beginJob() {
    if(state_ != sInit) return;
    bk::beginJob();
    // can only be run if in the initial state
    changeState(mBeginJob);

    // StateSentry toerror(this); // should we add this ? 
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);
    
    //NOTE:  This implementation assumes 'Job' means one call 
    // the EventProcessor::run
    // If it really means once per 'application' then this code will
    // have to be changed.
    // Also have to deal with case where have 'run' then new Module 
    // added and do 'run'
    // again.  In that case the newly added Module needs its 'beginJob'
    // to be called.
    EventSetup const& es =
      esp_->eventSetupForInstance(IOVSyncValue::beginOfTime());
    if(looper_) {
       looper_->beginOfJob(es);
    }
    try {
      input_->doBeginJob(es);
    } catch(cms::Exception& e) {
      LogError("BeginJob") << "A cms::Exception happened while processing the beginJob of the 'source'\n";
      e << "A cms::Exception happened while processing the beginJob of the 'source'\n";
      throw;
    } catch(std::exception& e) {
      LogError("BeginJob") << "A std::exception happened while processing the beginJob of the 'source'\n";
      throw;
    } catch(...) {
      LogError("BeginJob") << "An unknown exception happened while processing the beginJob of the 'source'\n";
      throw;
    }
    schedule_->beginJob(es);
    if (!allModuleNames().empty()) {
      cms::Exception exception("Modules still calling beginJob(EventSetup):\n");
      for (std::set<std::string>::const_iterator it = allModuleNames().begin(), itEnd = allModuleNames().end(); it != itEnd; ++it) {
	exception << *it << "\n";
      }
      throw exception;
    }
    actReg_->postBeginJobSignal_();
    // toerror.succeeded(); // should we add this?
  }

  void
  EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c;

    // only allowed to run if state is sIdle, sJobReady, sRunGiven
    c.call(boost::bind(&EventProcessor::changeState, this, mEndJob));

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);  

    c.call(boost::bind(&EventProcessor::terminateMachine, this));
    c.call(boost::bind(&Schedule::endJob, schedule_.get()));
    c.call(boost::bind(&InputSource::doEndJob, input_));
    if (looper_) {
      c.call(boost::bind(&EDLooper::endOfJob, looper_));
    }
    c.call(boost::bind(&ActivityRegistry::PostEndJob::operator(), &actReg_->postEndJobSignal_));
    if (c.hasThrown()) {
      c.rethrow();
    }
  }

  ServiceToken
  EventProcessor::getToken() {
    return serviceToken_;
  }
  
  //Setup signal handler to listen for when forked children stop
  namespace {
    volatile bool child_failed = false;
    volatile unsigned int num_children_done = 0;
    
    extern "C" {
      void ep_sigchld(int, siginfo_t*, void*) {
        //printf("in sigchld\n");
        //FDEBUG(1) << "in sigchld handler\n";
        int stat_loc;
        pid_t p = waitpid(-1, &stat_loc, WNOHANG); 
        while(0<p) {
          //printf("  looping\n");
          if(WIFEXITED(stat_loc)) {
            ++num_children_done;
            if(0 != WEXITSTATUS(stat_loc)) {
              child_failed = true;
            }
          }
          if(WIFSIGNALED(stat_loc)) {
            ++num_children_done;
            child_failed = true;
          }
          p = waitpid(-1, &stat_loc, WNOHANG); 
        }
      }
    }
    
  }
  
  enum {
    kChildSucceed,
    kChildExitBadly,
    kChildSegv,
    kMaxChildAction
  };
  
  bool 
  EventProcessor::forkProcess() {

    if(0 == numberOfForkedChildren_) {return true;}
    assert(0<numberOfForkedChildren_);
    //do what we want done in common
    {
      beginJob(); //make sure this was run
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      
      InputSource::ItemType itemType;
      itemType = input_->nextItemType();

      assert(itemType == InputSource::IsFile);
      {
        readFile();
      }
      itemType = input_->nextItemType();
      assert(itemType == InputSource::IsRun);
      
      int run = readAndCacheRun();
      
      RunPrincipal& runPrincipal = principalCache_.runPrincipal(run);
      std::cout <<" prefetching for run "<<runPrincipal.run()<<std::endl;
      IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0),
                      runPrincipal.beginTime());
      EventSetup const& es = esp_->eventSetupForInstance(ts);

      //now get all the data available in the EventSetup
      std::vector<eventsetup::EventSetupRecordKey> recordKeys;
      es.fillAvailableRecordKeys(recordKeys);
      std::vector<eventsetup::DataKey> dataKeys;
      for(std::vector<eventsetup::EventSetupRecordKey>::const_iterator itKey = recordKeys.begin(), itEnd = recordKeys.end();
          itKey != itEnd;
          ++itKey) {
        eventsetup::EventSetupRecord const* recordPtr = es.find(*itKey);
        //see if this is on our exclusion list
        ExcludedDataMap::const_iterator itExcludeRec = eventSetupDataToExcludeFromPrefetching_.find(itKey->type().name());
        const ExcludedData* excludedData(0);
        if(itExcludeRec != eventSetupDataToExcludeFromPrefetching_.end()) {
          excludedData=&(itExcludeRec->second);
          if (excludedData->size()==0 || excludedData->begin()->first=="*") {
            //skip all items in this record
            continue;
          }
        }
        if(0 != recordPtr) {
          dataKeys.clear();
          recordPtr->fillRegisteredDataKeys(dataKeys);
          for(std::vector<eventsetup::DataKey>::const_iterator itDataKey = dataKeys.begin(), itDataKeyEnd = dataKeys.end();
              itDataKey != itDataKeyEnd;
              ++itDataKey) {
            //std::cout <<"  "<<itDataKey->type().name()<<" "<<itDataKey->name().value()<<std::endl;
            if (0!=excludedData && excludedData->find(std::make_pair(itDataKey->type().name(),itDataKey->name().value()))!=excludedData->end()) {
              std::cout <<"   excluding:"<<itDataKey->type().name()<<" "<<itDataKey->name().value()<<std::endl;
              continue;
            }
            try {
              recordPtr->doGet(*itDataKey);
            } catch(cms::Exception& e) {
             edm::LogWarning("EventSetupPreFetching")<<e.what();
            }
          }
        }
      }
    }
    std::cout <<"  done prefetching"<<std::endl;

    //Now actually do the forking
    actReg_->preForkReleaseResourcesSignal_();
    input_->doPreForkReleaseResources();
    schedule_->preForkReleaseResources();

    installCustomHandler(SIGCHLD, ep_sigchld);
    unsigned int childIndex = 0;
    unsigned int const kMaxChildren = numberOfForkedChildren_;
    std::vector<pid_t> childrenIds;
    childrenIds.reserve(kMaxChildren);
    for(; childIndex < kMaxChildren; ++childIndex) {
      pid_t value = fork();
      if(value == 0) {
        std::cout << "I am child " << childIndex << " with pgid " << getpgrp() << std::endl;
        break;
      }
      if(value < 0) {
        std::cout << "failed to create a child" << std::endl;
        exit(-1);
      }
      childrenIds.push_back(value);
    }

    if(childIndex < kMaxChildren) {
      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      actReg_->postForkReacquireResourcesSignal_(childIndex, kMaxChildren);
      input_->doPostForkReacquireResources(childIndex, kMaxChildren, numberOfSequentialEventsPerChild_);
      schedule_->postForkReacquireResources(childIndex, kMaxChildren);
      //NOTE: sources have to reset themselves by listening to the post fork message
      //rewindInput();
      return true;
    }
    
    //this is the original which is now the master for all the children
    
    //Need to wait for signals from the children or externally
    // To wait we must
    // 1) block the signals we want to wait on so we do not have a race condition
    // 2) check that we haven't already meet our ending criteria
    // 3) call sigsuspend which unblocks the signals and waits until a signal is caught
    sigset_t blockingSigSet;
    sigset_t unblockingSigSet;
    sigset_t oldSigSet;
    pthread_sigmask(SIG_SETMASK, NULL, &unblockingSigSet);
    pthread_sigmask(SIG_SETMASK, NULL, &blockingSigSet);
    sigaddset(&blockingSigSet, SIGCHLD);
    sigaddset(&blockingSigSet, SIGUSR2);
    sigaddset(&blockingSigSet, SIGINT);
    sigdelset(&unblockingSigSet, SIGCHLD);
    sigdelset(&unblockingSigSet, SIGUSR2);
    sigdelset(&unblockingSigSet, SIGINT);
    pthread_sigmask(SIG_BLOCK, &blockingSigSet, &oldSigSet);
    while(!shutdown_flag && !child_failed && (childrenIds.size() != num_children_done)) {
      sigsuspend(&unblockingSigSet);
      std::cout << "woke from sigwait" << std::endl;
    }
    pthread_sigmask(SIG_SETMASK, &oldSigSet, NULL);
    
    std::cout << "num children who have already stopped " << num_children_done << std::endl;
    if(child_failed) {
      std::cout << "child failed" << std::endl;
    }
    if(shutdown_flag) {
      std::cout << "asked to shutdown" << std::endl;
    }
    if(shutdown_flag || (child_failed && (num_children_done != childrenIds.size()))) {
      std::cout << "must stop children" << std::endl;
      for(std::vector<pid_t>::iterator it = childrenIds.begin(), itEnd = childrenIds.end();
	  it != itEnd; ++it) {
	/* int result = */ kill(*it, SIGUSR2);
      }
      pthread_sigmask(SIG_BLOCK, &blockingSigSet, &oldSigSet);
      while(num_children_done != kMaxChildren) {
	sigsuspend(&unblockingSigSet);
      } 
      pthread_sigmask(SIG_SETMASK, &oldSigSet, NULL);
    }  
    return false;
  }

   
  void
  EventProcessor::connectSigs(EventProcessor* ep) {
    // When the FwkImpl signals are given, pass them to the
    // appropriate EventProcessor signals so that the outside world
    // can see the signal.
    actReg_->preProcessEventSignal_.connect(ep->preProcessEventSignal_);
    actReg_->postProcessEventSignal_.connect(ep->postProcessEventSignal_);
  }

  std::vector<ModuleDescription const*>
  EventProcessor::getAllModuleDescriptions() const {
    return schedule_->getAllModuleDescriptions();
  }

  int
  EventProcessor::totalEvents() const {
    return schedule_->totalEvents();
  }

  int
  EventProcessor::totalEventsPassed() const {
    return schedule_->totalEventsPassed();
  }

  int
  EventProcessor::totalEventsFailed() const {
    return schedule_->totalEventsFailed();
  }

  void 
  EventProcessor::enableEndPaths(bool active) {
    schedule_->enableEndPaths(active);
  }

  bool 
  EventProcessor::endPathsEnabled() const {
    return schedule_->endPathsEnabled();
  }
  
  void
  EventProcessor::getTriggerReport(TriggerReport& rep) const {
    schedule_->getTriggerReport(rep);
  }

  void
  EventProcessor::clearCounters() {
    schedule_->clearCounters();
  }


  char const* EventProcessor::currentStateName() const {
    return stateName(getState());
  }

  char const* EventProcessor::stateName(State s) const {
    return stateNames[s];
  }

  char const* EventProcessor::msgName(Msg m) const {
    return msgNames[m];
  }

  State EventProcessor::getState() const {
    return state_;
  }

  EventProcessor::StatusCode EventProcessor::statusAsync() const {
    // the thread will record exception/error status in the event processor
    // for us to look at and report here
    return last_rc_;
  }

  void
  EventProcessor::setRunNumber(RunNumber_t runNumber) {
    if (runNumber == 0) {
      runNumber = 1;
      LogWarning("Invalid Run")
        << "EventProcessor::setRunNumber was called with an invalid run number (0)\n"
	<< "Run number was set to 1 instead\n";
    }

    // inside of beginJob there is a check to see if it has been called before
    beginJob();
    changeState(mSetRun);

    // interface not correct yet
    input_->setRunNumber(runNumber);
  }

  void
  EventProcessor::declareRunNumber(RunNumber_t runNumber) {
    // inside of beginJob there is a check to see if it has been called before
    beginJob();
    changeState(mSetRun);

    // interface not correct yet - wait for Bill to be done with run/lumi loop stuff 21-Jun-2007
    //input_->declareRunNumber(runNumber);
  }

  EventProcessor::StatusCode 
  EventProcessor::waitForAsyncCompletion(unsigned int timeout_seconds) {
    bool rc = true;
    boost::xtime timeout;
    boost::xtime_get(&timeout, boost::TIME_UTC); 
    timeout.sec += timeout_seconds;

    // make sure to include a timeout here so we don't wait forever
    // I suspect there are still timing issues with thread startup
    // and the setting of the various control variables (stop_count, id_set)
    {
      boost::mutex::scoped_lock sl(stop_lock_);

      // look here - if runAsync not active, just return the last return code
      if(stop_count_ < 0) return last_rc_;

      if(timeout_seconds == 0)
	while(stop_count_ == 0) stopper_.wait(sl);
      else
	while(stop_count_ == 0 &&
	      (rc = stopper_.timed_wait(sl, timeout)) == true);
      
      if(rc == false) {
	  // timeout occurred
	  // if(id_set_) pthread_kill(event_loop_id_, my_sig_num_);
	  // this is a temporary hack until we get the input source
	  // upgraded to allow blocking input sources to be unblocked

	  // the next line is dangerous and causes all sorts of trouble
	  if(id_set_) pthread_cancel(event_loop_id_);

	  // we will not do anything yet
	  LogWarning("timeout")
	    << "An asynchronous request was made to shut down "
	    << "the event loop "
	    << "and the event loop did not shutdown after "
	    << timeout_seconds << " seconds\n";
      } else {
	  event_loop_->join();
	  event_loop_.reset();
	  id_set_ = false;
	  stop_count_ = -1;
      }
    }
    return rc == false ? epTimedOut : last_rc_;
  }

  EventProcessor::StatusCode 
  EventProcessor::waitTillDoneAsync(unsigned int timeout_value_secs) {
    StatusCode rc = waitForAsyncCompletion(timeout_value_secs);
    if(rc != epTimedOut) changeState(mCountComplete);
    else errorState();
    return rc;
  }

  
  EventProcessor::StatusCode EventProcessor::stopAsync(unsigned int secs) {
    changeState(mStopAsync);
    StatusCode rc = waitForAsyncCompletion(secs);
    if(rc != epTimedOut) changeState(mFinished);
    else errorState();
    return rc;
  }
  
  EventProcessor::StatusCode EventProcessor::shutdownAsync(unsigned int secs) {
    changeState(mShutdownAsync);
    StatusCode rc = waitForAsyncCompletion(secs);
    if(rc != epTimedOut) changeState(mFinished);
    else errorState();
    return rc;
  }
  
  void EventProcessor::errorState() {
    state_ = sError;
  }

  // next function irrelevant now
  EventProcessor::StatusCode EventProcessor::doneAsync(Msg m) {
    // make sure to include a timeout here so we don't wait forever
    // I suspect there are still timing issues with thread startup
    // and the setting of the various control variables (stop_count, id_set)
    changeState(m);
    return waitForAsyncCompletion(60*2);
  }
  
  void EventProcessor::changeState(Msg msg) {
    // most likely need to serialize access to this routine

    boost::mutex::scoped_lock sl(state_lock_);
    State curr = state_;
    int rc;
    // found if(not end of table) and 
    // (state == table.state && (msg == table.message || msg == any))
    for(rc = 0;
	table[rc].current != sInvalid && 
	  (curr != table[rc].current || 
	   (curr == table[rc].current && 
	     msg != table[rc].message && table[rc].message != mAny));
	++rc);

    if(table[rc].current == sInvalid)
      throw cms::Exception("BadState")
	<< "A member function of EventProcessor has been called in an"
	<< " inappropriate order.\n"
	<< "Bad transition from " << stateName(curr) << " "
	<< "using message " << msgName(msg) << "\n"
	<< "No where to go from here.\n";

    FDEBUG(1) << "changeState: current=" << stateName(curr)
	      << ", message=" << msgName(msg) 
	      << " -> new=" << stateName(table[rc].final) << "\n";

    state_ = table[rc].final;
  }

  void EventProcessor::runAsync() {
    using boost::thread;
    beginJob();
    {
      boost::mutex::scoped_lock sl(stop_lock_);
      if(id_set_ == true) {
	  std::string err("runAsync called while async event loop already running\n");
	  LogError("FwkJob") << err;
	  throw cms::Exception("BadState") << err;
      }

      changeState(mRunAsync);

      stop_count_ = 0;
      last_rc_ = epSuccess; // forget the last value!
      event_loop_.reset(new thread(boost::bind(EventProcessor::asyncRun, this)));
      boost::xtime timeout;
      boost::xtime_get(&timeout, boost::TIME_UTC); 
      timeout.sec += 60; // 60 seconds to start!!!!
      if(starter_.timed_wait(sl, timeout) == false) {
	  // yikes - the thread did not start
	  throw cms::Exception("BadState")
	    << "Async run thread did not start in 60 seconds\n";
      }
    }
  }

  void EventProcessor::asyncRun(EventProcessor* me) {
    // set up signals to allow for interruptions
    // ignore all other signals
    // make sure no exceptions escape out

    // temporary hack until we modify the input source to allow
    // wakeup calls from other threads.  This mimics the solution
    // in EventFilter/Processor, which I do not like.
    // allowing cancels means that the thread just disappears at
    // certain points.  This is bad for C++ stack variables.
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
    //pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, 0);
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);

    {
      boost::mutex::scoped_lock(me->stop_lock_);
      me->event_loop_id_ = pthread_self();
      me->id_set_ = true;
      me->starter_.notify_all();
    }

    Status rc = epException;
    FDEBUG(2) << "asyncRun starting ......................\n";

    try {
      bool onlineStateTransitions = true;
      rc = me->runToCompletion(onlineStateTransitions);
    }
    catch (cms::Exception& e) {
      LogError("FwkJob") << "cms::Exception caught in "
			      << "EventProcessor::asyncRun" 
			      << "\n"
			      << e.explainSelf();
      me->last_error_text_ = e.explainSelf();
    }
    catch (std::exception& e) {
      LogError("FwkJob") << "Standard library exception caught in " 
			      << "EventProcessor::asyncRun" 
			      << "\n"
			      << e.what();
      me->last_error_text_ = e.what();
    }
    catch (...) {
      LogError("FwkJob") << "Unknown exception caught in "
			      << "EventProcessor::asyncRun" 
			      << "\n";
      me->last_error_text_ = "Unknown exception caught";
      rc = epOther;
    }

    me->last_rc_ = rc;

    {
      // notify anyone waiting for exit that we are doing so now
      boost::mutex::scoped_lock sl(me->stop_lock_);
      ++me->stop_count_;
      me->stopper_.notify_all();
    }
    FDEBUG(2) << "asyncRun ending ......................\n";
  }


  EventProcessor::StatusCode
  EventProcessor::runToCompletion(bool onlineStateTransitions) {

    StateSentry toerror(this);

    int numberOfEventsToProcess = -1;
    StatusCode returnCode = runCommon(onlineStateTransitions, numberOfEventsToProcess);

    if (machine_.get() != 0) {
      throw edm::Exception(errors::LogicError)
        << "State machine not destroyed on exit from EventProcessor::runToCompletion\n"
	<< "Please report this error to the Framework group\n";
    }

    toerror.succeeded();

    return returnCode;
  }

  EventProcessor::StatusCode
  EventProcessor::runEventCount(int numberOfEventsToProcess) {

    StateSentry toerror(this);

    bool onlineStateTransitions = false;
    StatusCode returnCode = runCommon(onlineStateTransitions, numberOfEventsToProcess);

    toerror.succeeded();

    return returnCode;
  }

  EventProcessor::StatusCode
  EventProcessor::runCommon(bool onlineStateTransitions, int numberOfEventsToProcess) {

    // Reusable event principal
    boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_, *processConfiguration_));
    principalCache_.insert(ep);

    beginJob(); //make sure this was called

    if (!onlineStateTransitions) changeState(mRunCount);

    StatusCode returnCode = epSuccess;
    stateMachineWasInErrorState_ = false;

    // make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    if (machine_.get() == 0) {
 
      statemachine::FileMode fileMode;
      if (fileMode_.empty()) fileMode = statemachine::FULLMERGE;
      else if (fileMode_ == std::string("MERGE")) fileMode = statemachine::MERGE;
      else if (fileMode_ == std::string("NOMERGE")) fileMode = statemachine::NOMERGE;
      else if (fileMode_ == std::string("FULLMERGE")) fileMode = statemachine::FULLMERGE;
      else if (fileMode_ == std::string("FULLLUMIMERGE")) fileMode = statemachine::FULLLUMIMERGE;
      else {
 	throw edm::Exception(errors::Configuration, "Illegal fileMode parameter value: ")
	    << fileMode_ << ".\n"
	    << "Legal values are 'MERGE', 'NOMERGE', 'FULLMERGE', and 'FULLLUMIMERGE'.\n";
      }

      machine_.reset(new statemachine::Machine(this,
                                               fileMode,
                                               handleEmptyRuns_,
                                               handleEmptyLumis_));

      machine_->initiate();
    }

    try {

      InputSource::ItemType itemType;

      int iEvents = 0;

      while (true) {

        itemType = input_->nextItemType();

        FDEBUG(1) << "itemType = " << itemType << "\n";

        // These are used for asynchronous running only and
        // and are checking to see if stopAsync or shutdownAsync
        // were called from another thread.  In the future, we
        // may need to do something better than polling the state.
        // With the current code this is the simplest thing and
        // it should always work.  If the interaction between
        // threads becomes more complex this may cause problems.
        if (state_ == sStopping) {
          FDEBUG(1) << "In main processing loop, encountered sStopping state\n";
          forceLooperToEnd_ = true;
          machine_->process_event(statemachine::Stop());
          forceLooperToEnd_ = false;
          break;
        }
        else if (state_ == sShuttingDown) {
          FDEBUG(1) << "In main processing loop, encountered sShuttingDown state\n";
          forceLooperToEnd_ = true;
          machine_->process_event(statemachine::Stop());
          forceLooperToEnd_ = false;
          break;
        }

        // Look for a shutdown signal
        {
          boost::mutex::scoped_lock sl(usr2_lock);
          if (shutdown_flag) {
            changeState(mShutdownSignal);
            returnCode = epSignal;
            forceLooperToEnd_ = true;
            machine_->process_event(statemachine::Stop());
            forceLooperToEnd_ = false;
            break;
	  }
        }

        if (itemType == InputSource::IsStop) {
          machine_->process_event(statemachine::Stop());
        }
        else if (itemType == InputSource::IsFile) {
          machine_->process_event(statemachine::File());
        }
        else if (itemType == InputSource::IsRun) {
          machine_->process_event(statemachine::Run(input_->run()));
        }
        else if (itemType == InputSource::IsLumi) {
          machine_->process_event(statemachine::Lumi(input_->luminosityBlock()));
        }
        else if (itemType == InputSource::IsEvent) {
          machine_->process_event(statemachine::Event());
          ++iEvents;
          if (numberOfEventsToProcess > 0 && iEvents >= numberOfEventsToProcess) {
            returnCode = epCountComplete;            
            changeState(mInputExhausted);
            FDEBUG(1) << "Event count complete, pausing event loop\n";
            break;
          }
        }
        // This should be impossible
        else {
          throw edm::Exception(errors::LogicError)
	    << "Unknown next item type passed to EventProcessor\n"
	    << "Please report this error to the Framework group\n";
        }

        if (machine_->terminated()) {
          changeState(mInputExhausted);
          break;
        }
      }  // End of loop over state machine events
    } // Try block 

    // Some comments on exception handling related to the boost state machine:
    //
    // Some states used in the machine are special because they
    // perform actions while the machine is being terminated, actions
    // such as close files, call endRun, call endLumi etc ...  Each of these
    // states has two functions that perform these actions.  The functions
    // are almost identical.  The major difference is that one version
    // catches all exceptions and the other lets exceptions pass through.
    // The destructor catches them and the other function named "exit" lets
    // them pass through.  On a normal termination, boost will always call
    // "exit" and then the state destructor.  In our state classes, the
    // the destructors do nothing if the exit function already took
    // care of things.  Here's the interesting part.  When boost is
    // handling an exception the "exit" function is not called (a boost
    // feature).
    //
    // If an exception occurs while the boost machine is in control
    // (which usually means inside a process_event call), then
    // the boost state machine destroys its states and "terminates" itself.
    // This already done before we hit the catch blocks below. In this case
    // the call to terminateMachine below only destroys an already
    // terminated state machine.  Because exit is not called, the state destructors
    // handle cleaning up lumis, runs, and files.  The destructors swallow
    // all exceptions and only pass through the exceptions messages which
    // are tacked onto the original exception below.
    // 
    // If an exception occurs when the boost state machine is not
    // in control (outside the process_event functions), then boost
    // cannot destroy its own states.  The terminateMachine function
    // below takes care of that.  The flag "alreadyHandlingException"
    // is set true so that the state exit functions do nothing (and
    // cannot throw more exceptions while handling the first).  Then the
    // state destructors take care of this because exit did nothing.
    //
    // In both cases above, the EventProcessor::endOfLoop function is
    // not called because it can throw exceptions.
    //
    // One tricky aspect of the state machine is that things which can
    // throw should not be invoked by the state machine while another
    // exception is being handled.
    // Another tricky aspect is that it appears to be important to 
    // terminate the state machine before invoking its destructor.
    // We've seen crashes which are not understood when that is not
    // done.  Maintainers of this code should be careful about this.

    catch (cms::Exception& e) {
      alreadyHandlingException_ = true;
      terminateMachine();
      alreadyHandlingException_ = false;
      e << "cms::Exception caught in EventProcessor and rethrown\n";
      e << exceptionMessageLumis_;
      e << exceptionMessageRuns_;
      e << exceptionMessageFiles_;
      throw e;
    }
    catch (std::bad_alloc& e) {
      alreadyHandlingException_ = true;
      terminateMachine();
      alreadyHandlingException_ = false;
      throw cms::Exception("std::bad_alloc")
        << "The EventProcessor caught a std::bad_alloc exception and converted it to a cms::Exception\n"
        << "The job has probably exhausted the virtual memory available to the process.\n"
        << exceptionMessageLumis_
        << exceptionMessageRuns_
        << exceptionMessageFiles_;
    }
    catch (std::exception& e) {
      alreadyHandlingException_ = true;
      terminateMachine();
      alreadyHandlingException_ = false;
      throw cms::Exception("StdException")
        << "The EventProcessor caught a std::exception and converted it to a cms::Exception\n"
        << "Previous information:\n" << e.what() << "\n"
        << exceptionMessageLumis_
        << exceptionMessageRuns_
        << exceptionMessageFiles_;
    }
    catch (...) {
      alreadyHandlingException_ = true;
      terminateMachine();
      alreadyHandlingException_ = false;
      throw cms::Exception("Unknown")
        << "The EventProcessor caught an unknown exception type and converted it to a cms::Exception\n"
        << exceptionMessageLumis_
        << exceptionMessageRuns_
        << exceptionMessageFiles_;
    }

    if (machine_->terminated()) {
      FDEBUG(1) << "The state machine reports it has been terminated\n";
      machine_.reset();
    }

    if (!onlineStateTransitions) changeState(mFinished);

    if (stateMachineWasInErrorState_) {
      throw cms::Exception("BadState")
	<< "The boost state machine in the EventProcessor exited after\n"
	<< "entering the Error state.\n";
    }

    return returnCode;
  }

  void EventProcessor::readFile() {
    FDEBUG(1) << " \treadFile\n";
    fb_ = input_->readFile();
    if (numberOfForkedChildren_ > 0) {
	fb_->setNotFastClonable(FileBlock::ParallelProcesses);
    }
  }

  void EventProcessor::closeInputFile() {
    input_->closeFile(fb_);
    FDEBUG(1) << "\tcloseInputFile\n";
  }

  void EventProcessor::openOutputFiles() {
    schedule_->openOutputFiles(*fb_);
    FDEBUG(1) << "\topenOutputFiles\n";
  }

  void EventProcessor::closeOutputFiles() {
    schedule_->closeOutputFiles();
    FDEBUG(1) << "\tcloseOutputFiles\n";
  }

  void EventProcessor::respondToOpenInputFile() {
    schedule_->respondToOpenInputFile(*fb_);
    FDEBUG(1) << "\trespondToOpenInputFile\n";
  }

  void EventProcessor::respondToCloseInputFile() {
    schedule_->respondToCloseInputFile(*fb_);
    FDEBUG(1) << "\trespondToCloseInputFile\n";
  }

  void EventProcessor::respondToOpenOutputFiles() {
    schedule_->respondToOpenOutputFiles(*fb_);
    FDEBUG(1) << "\trespondToOpenOutputFiles\n";
  }

  void EventProcessor::respondToCloseOutputFiles() {
    schedule_->respondToCloseOutputFiles(*fb_);
    FDEBUG(1) << "\trespondToCloseOutputFiles\n";
  }

  void EventProcessor::startingNewLoop() {
    shouldWeStop_ = false;
    if (looper_) {
      looper_->doStartingNewLoop();
    }
    FDEBUG(1) << "\tstartingNewLoop\n";
  }

  bool EventProcessor::endOfLoop() {
    if (looper_) {
      EDLooper::Status status = looper_->doEndOfLoop(esp_->eventSetup());
      if (status != EDLooper::kContinue || forceLooperToEnd_) return true;
      else return false;
    }
    FDEBUG(1) << "\tendOfLoop\n";
    return true;
  }

  void EventProcessor::rewindInput() {
    input_->repeat();
    input_->rewind();
    FDEBUG(1) << "\trewind\n";
  }

  void EventProcessor::prepareForNextLoop() {
    looper_->prepareForNextLoop(esp_.get());
    FDEBUG(1) << "\tprepareForNextLoop\n";
  }

  void EventProcessor::writeLumiCache() {
    while (!principalCache_.noMoreLumis()) {
      schedule_->writeLumi(principalCache_.lowestLumi());
      principalCache_.deleteLowestLumi();      
    }
    input_->respondToClearingLumiCache();
    FDEBUG(1) << "\twriteLumiCache\n";
  }

  void EventProcessor::writeRunCache() {
    while (!principalCache_.noMoreRuns()) {
      schedule_->writeRun(principalCache_.lowestRun());
      principalCache_.deleteLowestRun();      
    }
    input_->respondToClearingRunCache();
    FDEBUG(1) << "\twriteRunCache\n";
  }

  bool EventProcessor::shouldWeCloseOutput() const {
    FDEBUG(1) << "\tshouldWeCloseOutput\n";
    return schedule_->shouldWeCloseOutput();
  }

  void EventProcessor::doErrorStuff() {
    FDEBUG(1) << "\tdoErrorStuff\n";
    LogError("StateMachine")
      << "The EventProcessor state machine encountered an unexpected event\n"
      << "and went to the error state\n"
      << "Will attempt to terminate processing normally\n"
      << "(IF using the looper the next loop will be attempted)\n"
      << "This likely indicates a bug in an input module or corrupted input or both\n";
    stateMachineWasInErrorState_ = true;
  }

  void EventProcessor::beginRun(int run) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(run);
    input_->doBeginRun(runPrincipal);
    IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0),
                    runPrincipal.beginTime());
    EventSetup const& es = esp_->eventSetupForInstance(ts);
    schedule_->processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionBegin> >(runPrincipal, es);
    FDEBUG(1) << "\tbeginRun " << run << "\n";
    if (looper_) {
      looper_->doBeginRun(runPrincipal, es);
    }
  }

  void EventProcessor::endRun(int run) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(run);
    input_->doEndRun(runPrincipal);
    IOVSyncValue ts(EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
                    runPrincipal.endTime());
    EventSetup const& es = esp_->eventSetupForInstance(ts);
    schedule_->processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionEnd> >(runPrincipal, es);
    FDEBUG(1) << "\tendRun " << run << "\n";
    if (looper_) {
      looper_->doEndRun(runPrincipal, es);
    }
  }

  void EventProcessor::beginLumi(int run, int lumi) {
    LuminosityBlockPrincipal& lumiPrincipal = principalCache_.lumiPrincipal(run, lumi);
    input_->doBeginLumi(lumiPrincipal);
    // NOTE: Using 0 as the event number for the begin of a lumi block is a bad idea
    // lumi blocks know their start and end times why not also start and end events?
    IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), 0), lumiPrincipal.beginTime());
    EventSetup const& es = esp_->eventSetupForInstance(ts);
    schedule_->processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionBegin> >(lumiPrincipal, es);
    FDEBUG(1) << "\tbeginLumi " << run << "/" << lumi << "\n";
    if (looper_) {
      looper_->doBeginLuminosityBlock(lumiPrincipal, es);
    }
  }

  void EventProcessor::endLumi(int run, int lumi) {
    LuminosityBlockPrincipal& lumiPrincipal = principalCache_.lumiPrincipal(run, lumi);
    input_->doEndLumi(lumiPrincipal);
    //NOTE: Using the max event number for the end of a lumi block is a bad idea
    // lumi blocks know their start and end times why not also start and end events?
    IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), EventID::maxEventNumber()),
                    lumiPrincipal.endTime());
    EventSetup const& es = esp_->eventSetupForInstance(ts);
    schedule_->processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionEnd> >(lumiPrincipal, es);
    FDEBUG(1) << "\tendLumi " << run << "/" << lumi << "\n";
    if (looper_) {
      looper_->doEndLuminosityBlock(lumiPrincipal, es);
    }
  }

  int EventProcessor::readAndCacheRun() {
    input_->readAndCacheRun();
    return input_->markRun();
  }

  int EventProcessor::readAndCacheLumi() {
    input_->readAndCacheLumi();
    return input_->markLumi();
  }

  void EventProcessor::writeRun(int run) {
    schedule_->writeRun(principalCache_.runPrincipal(run));
    FDEBUG(1) << "\twriteRun " << run << "\n";
  }

  void EventProcessor::deleteRunFromCache(int run) {
    principalCache_.deleteRun(run);
    FDEBUG(1) << "\tdeleteRunFromCache " << run << "\n";
  }

  void EventProcessor::writeLumi(int run, int lumi) {
    schedule_->writeLumi(principalCache_.lumiPrincipal(run, lumi));
    FDEBUG(1) << "\twriteLumi " << run << "/" << lumi << "\n";
  }

  void EventProcessor::deleteLumiFromCache(int run, int lumi) {
    principalCache_.deleteLumi(run, lumi);
    FDEBUG(1) << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  void EventProcessor::readAndProcessEvent() {
    EventPrincipal *pep = 0;
    try {
      pep = input_->readEvent(principalCache_.lumiPrincipalPtr());
      FDEBUG(1) << "\treadEvent\n";
    }
    catch(cms::Exception& e) {
      actions::ActionCodes action = act_table_.find(e.rootCause());
      if (action == actions::Rethrow) {
 	throw;
      } else {
        LogWarning(e.category())
          << "an exception occurred and all paths for the event are being skipped: \n"
          << e.what();
        return;
      }
    }
    assert(pep != 0);

    IOVSyncValue ts(pep->id(), pep->time());
    EventSetup const& es = esp_->eventSetupForInstance(ts);
    schedule_->processOneOccurrence<OccurrenceTraits<EventPrincipal, BranchActionBegin> >(*pep, es);
 
    if (looper_) {
      EDLooper::Status status = looper_->doDuringLoop(*pep, esp_->eventSetup());
      if (status != EDLooper::kContinue) shouldWeStop_ = true;
    }

    FDEBUG(1) << "\tprocessEvent\n";
    pep->clearEventPrincipal();
  }

  bool EventProcessor::shouldWeStop() const {
    FDEBUG(1) << "\tshouldWeStop\n";
    if (shouldWeStop_) return true;
    return schedule_->terminate();
  }

  void EventProcessor::setExceptionMessageFiles(std::string& message) {
    exceptionMessageFiles_ = message;
  }

  void EventProcessor::setExceptionMessageRuns(std::string& message) {
    exceptionMessageRuns_ = message;
  }

  void EventProcessor::setExceptionMessageLumis(std::string& message) {
    exceptionMessageLumis_ = message;
  }

  bool EventProcessor::alreadyHandlingException() const {
    return alreadyHandlingException_;
  }

  void EventProcessor::terminateMachine() {
    if (machine_.get() != 0) {
      if (!machine_->terminated()) {
        forceLooperToEnd_ = true;
        machine_->process_event(statemachine::Stop());
        forceLooperToEnd_ = false;
      }
      else {
        FDEBUG(1) << "EventProcess::terminateMachine  The state machine was already terminated \n";
      }
      if (machine_->terminated()) {
        FDEBUG(1) << "The state machine reports it has been terminated (3)\n";
      }
      machine_.reset();
    }
  }
}
