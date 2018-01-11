#include "FWCore/Framework/interface/EventProcessor.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"

#include "FWCore/Framework/interface/CommonParams.h"
#include "FWCore/Framework/interface/EDLooperBase.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/src/Breakpoints.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/src/SharedResourcesRegistry.h"
#include "FWCore/Framework/src/streamTransitionAsync.h"
#include "FWCore/Framework/src/globalTransitionAsync.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RootHandlers.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "MessageForSource.h"
#include "MessageForParent.h"
#include "LuminosityBlockProcessingStatus.h"

#include "boost/range/adaptor/reversed.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <utility>
#include <sstream>

#include <sys/ipc.h>
#include <sys/msg.h>

#include "tbb/task.h"

//Used for CPU affinity
#ifndef __APPLE__
#include <sched.h>
#endif

namespace {
  //Sentry class to only send a signal if an
  // exception occurs. An exception is identified
  // by the destructor being called without first
  // calling completedSuccessfully().
  class SendSourceTerminationSignalIfException {
  public:
    SendSourceTerminationSignalIfException(edm::ActivityRegistry* iReg):
      reg_(iReg) {}
    ~SendSourceTerminationSignalIfException() {
      if(reg_) {
        reg_->preSourceEarlyTerminationSignal_(edm::TerminationOrigin::ExceptionFromThisContext);
      }
    }
    void completedSuccessfully() {
      reg_ = nullptr;
    }
  private:
    edm::ActivityRegistry* reg_; // We do not use propagate_const because the registry itself is mutable.
  };

}

namespace edm {

  // ---------------------------------------------------------------
  std::unique_ptr<InputSource>
  makeInput(ParameterSet& params,
            CommonParams const& common,
            std::shared_ptr<ProductRegistry> preg,
            std::shared_ptr<BranchIDListHelper> branchIDListHelper,
            std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
            std::shared_ptr<ActivityRegistry> areg,
            std::shared_ptr<ProcessConfiguration const> processConfiguration,
            PreallocationConfiguration const& allocations) {
    ParameterSet* main_input = params.getPSetForUpdate("@main_input");
    if(main_input == nullptr) {
      throw Exception(errors::Configuration)
        << "There must be exactly one source in the configuration.\n"
        << "It is missing (or there are sufficient syntax errors such that it is not recognized as the source)\n";
    }

    std::string modtype(main_input->getParameter<std::string>("@module_type"));

    std::unique_ptr<ParameterSetDescriptionFillerBase> filler(
                                                              ParameterSetDescriptionFillerPluginFactory::get()->create(modtype));
    ConfigurationDescriptions descriptions(filler->baseType(), modtype);
    filler->fill(descriptions);

    try {
      convertException::wrap([&]() {
          descriptions.validate(*main_input, std::string("source"));
        });
    }
    catch (cms::Exception & iException) {
      std::ostringstream ost;
      ost << "Validating configuration of input source of type " << modtype;
      iException.addContext(ost.str());
      throw;
    }

    main_input->registerIt();

    // Fill in "ModuleDescription", in case the input source produces
    // any EDProducts, which would be registered in the ProductRegistry.
    // Also fill in the process history item for this process.
    // There is no module label for the unnamed input source, so
    // just use "source".
    // Only the tracked parameters belong in the process configuration.
    ModuleDescription md(main_input->id(),
                         main_input->getParameter<std::string>("@module_type"),
                         "source",
                         processConfiguration.get(),
                         ModuleDescription::getUniqueID());

    InputSourceDescription isdesc(md, preg, branchIDListHelper, thinnedAssociationsHelper, areg,
                                  common.maxEventsInput_, common.maxLumisInput_,
                                  common.maxSecondsUntilRampdown_, allocations);

    areg->preSourceConstructionSignal_(md);
    std::unique_ptr<InputSource> input;
    try {
      //even if we have an exception, send the signal
      std::shared_ptr<int> sentry(nullptr,[areg,&md](void*){areg->postSourceConstructionSignal_(md);});
      convertException::wrap([&]() {
          input = std::unique_ptr<InputSource>(InputSourceFactory::get()->makeInputSource(*main_input, isdesc).release());
          input->preEventReadFromSourceSignal_.connect(std::cref(areg->preEventReadFromSourceSignal_));
          input->postEventReadFromSourceSignal_.connect(std::cref(areg->postEventReadFromSourceSignal_));
        });
    }
    catch (cms::Exception& iException) {
      std::ostringstream ost;
      ost << "Constructing input source of type " << modtype;
      iException.addContext(ost.str());
      throw;
    }
    return input;
  }

  // ---------------------------------------------------------------
  std::shared_ptr<EDLooperBase>
  fillLooper(eventsetup::EventSetupsController& esController,
             eventsetup::EventSetupProvider& cp,
             ParameterSet& params) {
    std::shared_ptr<EDLooperBase> vLooper;

    std::vector<std::string> loopers = params.getParameter<std::vector<std::string> >("@all_loopers");

    if(loopers.empty()) {
      return vLooper;
    }

    assert(1 == loopers.size());

    for(std::vector<std::string>::iterator itName = loopers.begin(), itNameEnd = loopers.end();
        itName != itNameEnd;
        ++itName) {

      ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      vLooper = eventsetup::LooperFactory::get()->addTo(esController,
                                                        cp,
                                                        *providerPSet);
    }
    return vLooper;
  }

  // ---------------------------------------------------------------
  EventProcessor::EventProcessor(std::string const& config,
                                 ServiceToken const& iToken,
                                 serviceregistry::ServiceLegacy iLegacy,
                                 std::vector<std::string> const& defaultServices,
                                 std::vector<std::string> const& forcedServices) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcesses_(),
    historyAppender_(new HistoryAppender),
    fb_(),
    looper_(),
    deferredExceptionPtrIsSet_(false),
    sourceResourcesAcquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first),
    sourceMutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    fileModeNoMerge_(false),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    eventSetupDataToExcludeFromPrefetching_() {
    std::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
    auto processDesc = std::make_shared<ProcessDesc>(parameterSet);
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, iToken, iLegacy);
  }

  EventProcessor::EventProcessor(std::string const& config,
                                 std::vector<std::string> const& defaultServices,
                                 std::vector<std::string> const& forcedServices) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcesses_(),
    historyAppender_(new HistoryAppender),
    fb_(),
    looper_(),
    deferredExceptionPtrIsSet_(false),
    sourceResourcesAcquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first),
    sourceMutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    fileModeNoMerge_(false),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    asyncStopRequestedWhileProcessingEvents_(false),
    nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
  {
    std::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
    auto processDesc = std::make_shared<ProcessDesc>(parameterSet);
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
  }

  EventProcessor::EventProcessor(std::shared_ptr<ProcessDesc> processDesc,
                                 ServiceToken const& token,
                                 serviceregistry::ServiceLegacy legacy) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcesses_(),
    historyAppender_(new HistoryAppender),
    fb_(),
    looper_(),
    deferredExceptionPtrIsSet_(false),
    sourceResourcesAcquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first),
    sourceMutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    fileModeNoMerge_(false),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    asyncStopRequestedWhileProcessingEvents_(false),
    nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
  {
    init(processDesc, token, legacy);
  }


  EventProcessor::EventProcessor(std::string const& config, bool isPython):
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcesses_(),
    historyAppender_(new HistoryAppender),
    fb_(),
    looper_(),
    deferredExceptionPtrIsSet_(false),
    sourceResourcesAcquirer_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().first),
    sourceMutex_(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader().second),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    fileModeNoMerge_(false),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    asyncStopRequestedWhileProcessingEvents_(false),
    nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
  {
    if(isPython) {
      std::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
      auto processDesc = std::make_shared<ProcessDesc>(parameterSet);
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
    else {
      auto processDesc = std::make_shared<ProcessDesc>(config);
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
  }

  void
  EventProcessor::init(std::shared_ptr<ProcessDesc>& processDesc,
                       ServiceToken const& iToken,
                       serviceregistry::ServiceLegacy iLegacy) {

    //std::cerr << processDesc->dump() << std::endl;

    // register the empty parentage vector , once and for all
    ParentageRegistry::instance()->insertMapped(Parentage());

    // register the empty parameter set, once and for all.
    ParameterSet().registerIt();

    std::shared_ptr<ParameterSet> parameterSet = processDesc->getProcessPSet();

    // If there are subprocesses, pop the subprocess parameter sets out of the process parameter set
    auto subProcessVParameterSet = popSubProcessVParameterSet(*parameterSet);
    bool const hasSubProcesses = !subProcessVParameterSet.empty();

    // Now set some parameters specific to the main process.
    ParameterSet const& optionsPset(parameterSet->getUntrackedParameterSet("options", ParameterSet()));
    auto const& fileMode = optionsPset.getUntrackedParameter<std::string>("fileMode", "FULLMERGE");
    if(fileMode != "NOMERGE" and fileMode != "FULLMERGE") {
        throw Exception(errors::Configuration, "Illegal fileMode parameter value: ")
        << fileMode << ".\n"
        << "Legal values are 'NOMERGE' and 'FULLMERGE'.\n";
    } else {
      fileModeNoMerge_ = (fileMode == "NOMERGE");
    }
    forceESCacheClearOnNewRun_ = optionsPset.getUntrackedParameter<bool>("forceEventSetupCacheClearOnNewRun", false);
    //threading
    unsigned int nThreads=1;
    if(optionsPset.existsAs<unsigned int>("numberOfThreads",false)) {
      nThreads = optionsPset.getUntrackedParameter<unsigned int>("numberOfThreads");
      if(nThreads == 0) {
        nThreads = 1;
      }
    }
    /* TODO: when we support having each stream run in a different thread use this default
       unsigned int nStreams =nThreads;
    */
    unsigned int nStreams =1;
    if(optionsPset.existsAs<unsigned int>("numberOfStreams",false)) {
      nStreams = optionsPset.getUntrackedParameter<unsigned int>("numberOfStreams");
      if(nStreams==0) {
        nStreams = nThreads;
      }
    }
    if(nThreads >1) {
      edm::LogInfo("ThreadStreamSetup") <<"setting # threads "<<nThreads<<"\nsetting # streams "<<nStreams;
    }

    /*
      bool nRunsSet = false;
    */
    unsigned int nConcurrentRuns =1;
    /*
      if(nRunsSet = optionsPset.existsAs<unsigned int>("numberOfConcurrentRuns",false)) {
      nConcurrentRuns = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentRuns");
      }
    */
    unsigned int nConcurrentLumis =1;
    /*
      if(optionsPset.existsAs<unsigned int>("numberOfConcurrentLuminosityBlocks",false)) {
      nConcurrentLumis = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentLuminosityBlocks");
      } else {
      nConcurrentLumis = nConcurrentRuns;
      }
    */
    //Check that relationships between threading parameters makes sense
    /*
      if(nThreads<nStreams) {
      //bad
      }
      if(nConcurrentRuns>nStreams) {
      //bad
      }
      if(nConcurrentRuns>nConcurrentLumis) {
      //bad
      }
    */
    IllegalParameters::setThrowAnException(optionsPset.getUntrackedParameter<bool>("throwIfIllegalParameter", true));

    printDependencies_ =  optionsPset.getUntrackedParameter("printDependencies", false);

    // Now do general initialization
    ScheduleItems items;

    //initialize the services
    auto& serviceSets = processDesc->getServicesPSets();
    ServiceToken token = items.initServices(serviceSets, *parameterSet, iToken, iLegacy, true);
    serviceToken_ = items.addCPRandTNS(*parameterSet, token);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    if(nStreams>1) {
      edm::Service<RootHandlers> handler;
      handler->willBeUsingThreads();
    }

    // intialize miscellaneous items
    std::shared_ptr<CommonParams> common(items.initMisc(*parameterSet));

    // intialize the event setup provider
    esp_ = espController_->makeProvider(*parameterSet);

    // initialize the looper, if any
    looper_ = fillLooper(*espController_, *esp_, *parameterSet);
    if(looper_) {
      looper_->setActionTable(items.act_table_.get());
      looper_->attachTo(*items.actReg_);

      //For now loopers make us run only 1 transition at a time
      nStreams=1;
      nConcurrentLumis=1;
      nConcurrentRuns=1;
    }

    preallocations_ = PreallocationConfiguration{nThreads,nStreams,nConcurrentLumis,nConcurrentRuns};

    lumiQueue_ = std::make_unique<LimitedTaskQueue>(nConcurrentLumis);
    streamQueues_.resize(nStreams);
    streamLumiStatus_.resize(nStreams);
    
    // initialize the input source
    input_ = makeInput(*parameterSet,
                       *common,
                       items.preg(),
                       items.branchIDListHelper(),
                       items.thinnedAssociationsHelper(),
                       items.actReg_,
                       items.processConfiguration(),
                       preallocations_);

    // intialize the Schedule
    schedule_ = items.initSchedule(*parameterSet,hasSubProcesses,preallocations_,&processContext_);

    // set the data members
    act_table_ = std::move(items.act_table_);
    actReg_ = items.actReg_;
    preg_ = items.preg();
    branchIDListHelper_ = items.branchIDListHelper();
    thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
    processConfiguration_ = items.processConfiguration();
    processContext_.setProcessConfiguration(processConfiguration_.get());
    principalCache_.setProcessHistoryRegistry(input_->processHistoryRegistry());

    FDEBUG(2) << parameterSet << std::endl;

    principalCache_.setNumberOfConcurrentPrincipals(preallocations_);
    for(unsigned int index = 0; index<preallocations_.numberOfStreams(); ++index ) {
      // Reusable event principal
      auto ep = std::make_shared<EventPrincipal>(preg(), branchIDListHelper(),
                                                 thinnedAssociationsHelper(), *processConfiguration_, historyAppender_.get(), index);
      principalCache_.insert(std::move(ep));
    }
    
    for(unsigned int index =0; index < preallocations_.numberOfLuminosityBlocks(); ++index) {
      auto lp = std::make_unique<LuminosityBlockPrincipal>(preg(), *processConfiguration_,
                                                           historyAppender_.get(), index);
      principalCache_.insert(std::move(lp));
    }

    // fill the subprocesses, if there are any
    subProcesses_.reserve(subProcessVParameterSet.size());
    for(auto& subProcessPSet : subProcessVParameterSet) {
      subProcesses_.emplace_back(subProcessPSet,
                                 *parameterSet,
                                 preg(),
                                 branchIDListHelper(),
                                 *thinnedAssociationsHelper_,
                                 SubProcessParentageHelper(),
                                 *espController_,
                                 *actReg_,
                                 token,
                                 serviceregistry::kConfigurationOverrides,
                                 preallocations_,
                                 &processContext_);
    }
  }

  EventProcessor::~EventProcessor() {
    // Make the services available while everything is being deleted.
    ServiceToken token = getToken();
    ServiceRegistry::Operate op(token);

    // manually destroy all these thing that may need the services around
    // propagate_const<T> has no reset() function
    espController_ = nullptr;
    esp_ = nullptr;
    schedule_ = nullptr;
    input_ = nullptr;
    looper_ = nullptr;
    actReg_ = nullptr;

    pset::Registry::instance()->clear();
    ParentageRegistry::instance()->clear();
  }

  void
  EventProcessor::beginJob() {
    if(beginJobCalled_) return;
    beginJobCalled_=true;
    bk::beginJob();

    // StateSentry toerror(this); // should we add this ?
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    service::SystemBounds bounds(preallocations_.numberOfStreams(),
                                 preallocations_.numberOfLuminosityBlocks(),
                                 preallocations_.numberOfRuns(),
                                 preallocations_.numberOfThreads());
    actReg_->preallocateSignal_(bounds);
    schedule_->convertCurrentProcessAlias(processConfiguration_->processName());
    pathsAndConsumesOfModules_.initialize(schedule_.get(), preg());

    //NOTE: this may throw
    checkForModuleDependencyCorrectness(pathsAndConsumesOfModules_, printDependencies_);
    actReg_->preBeginJobSignal_(pathsAndConsumesOfModules_, processContext_);

    //NOTE:  This implementation assumes 'Job' means one call
    // the EventProcessor::run
    // If it really means once per 'application' then this code will
    // have to be changed.
    // Also have to deal with case where have 'run' then new Module
    // added and do 'run'
    // again.  In that case the newly added Module needs its 'beginJob'
    // to be called.

    //NOTE: in future we should have a beginOfJob for looper that takes no arguments
    //  For now we delay calling beginOfJob until first beginOfRun
    //if(looper_) {
    //   looper_->beginOfJob(es);
    //}
    try {
      convertException::wrap([&]() {
          input_->doBeginJob();
        });
    }
    catch(cms::Exception& ex) {
      ex.addContext("Calling beginJob for the source");
      throw;
    }
    schedule_->beginJob(*preg_);
    // toerror.succeeded(); // should we add this?
    for_all(subProcesses_, [](auto& subProcess){ subProcess.doBeginJob(); });
    actReg_->postBeginJobSignal_();

    for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
      schedule_->beginStream(i);
      for_all(subProcesses_, [i](auto& subProcess){ subProcess.doBeginStream(i); });
    }
  }

  void
  EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c("Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    //NOTE: this really should go elsewhere in the future
    for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
      c.call([this,i](){this->schedule_->endStream(i);});
      for(auto& subProcess : subProcesses_) {
        c.call([&subProcess,i](){ subProcess.doEndStream(i); } );
      }
    }
    auto actReg = actReg_.get();
    c.call([actReg](){actReg->preEndJobSignal_();});
    schedule_->endJob(c);
    for(auto& subProcess : subProcesses_) {
      c.call(std::bind(&SubProcess::doEndJob, &subProcess));
    }
    c.call(std::bind(&InputSource::doEndJob, input_.get()));
    if(looper_) {
      c.call(std::bind(&EDLooperBase::endOfJob, looper()));
    }
    c.call([actReg](){actReg->postEndJobSignal_();});
    if(c.hasThrown()) {
      c.rethrow();
    }
  }

  ServiceToken
  EventProcessor::getToken() {
    return serviceToken_;
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

  namespace {
#include "TransitionProcessors.icc"
  }

  bool
  EventProcessor::checkForAsyncStopRequest(StatusCode& returnCode) {
    bool returnValue = false;

    // Look for a shutdown signal
    if(shutdown_flag.load(std::memory_order_acquire)) {
      returnValue = true;
      returnCode = epSignal;
    }
    return returnValue;
  }

  InputSource::ItemType
  EventProcessor::nextTransitionType() {
    SendSourceTerminationSignalIfException sentry(actReg_.get());
    InputSource::ItemType itemType;
    //For now, do nothing with InputSource::IsSynchronize
    do {
      itemType = input_->nextItemType();
    } while( itemType == InputSource::IsSynchronize);
    
    sentry.completedSuccessfully();
    
    StatusCode returnCode=epSuccess;
    
    if(checkForAsyncStopRequest(returnCode)) {
      actReg_->preSourceEarlyTerminationSignal_(TerminationOrigin::ExternalSignal);
      return InputSource::IsStop;
    }

    return itemType;
  }

  std::pair<edm::ProcessHistoryID, edm::RunNumber_t>
  EventProcessor::nextRunID() {
    return std::make_pair(input_->reducedProcessHistoryID(), input_->run());
  }
  
  edm::LuminosityBlockNumber_t
  EventProcessor::nextLuminosityBlockID() {
    return input_->luminosityBlock();
  }

  EventProcessor::StatusCode
  EventProcessor::runToCompletion() {
    
    StatusCode returnCode=epSuccess;
    asyncStopStatusCodeFromProcessingEvents_=epSuccess;
    {
      beginJob(); //make sure this was called
      
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      
      nextItemTypeFromProcessingEvents_=InputSource::IsEvent;
      asyncStopRequestedWhileProcessingEvents_=false;
      try {
        FilesProcessor fp(fileModeNoMerge_);

        convertException::wrap([&]() {
          bool firstTime = true;
          do {
            if(not firstTime) {
              prepareForNextLoop();
              rewindInput();
            } else {
              firstTime = false;
            }
            startingNewLoop();
            
            auto trans = fp.processFiles(*this);
            
            fp.normalEnd();
            
            if(deferredExceptionPtrIsSet_.load()) {
              std::rethrow_exception(deferredExceptionPtr_);
            }
            if(trans != InputSource::IsStop) {
              //problem with the source
              doErrorStuff();
              
              throw cms::Exception("BadTransition")
              << "Unexpected transition change "
              << trans;

            }
          } while(not endOfLoop());
        }); // convertException::wrap
        
      } // Try block
      catch (cms::Exception & e) {
        if (!exceptionMessageLumis_.empty()) {
          e.addAdditionalInfo(exceptionMessageLumis_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageLumis_;
          }
        }
        if (!exceptionMessageRuns_.empty()) {
          e.addAdditionalInfo(exceptionMessageRuns_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageRuns_;
          }
        }
        if (!exceptionMessageFiles_.empty()) {
          e.addAdditionalInfo(exceptionMessageFiles_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageFiles_;
          }
        }
        throw;
      }
    }
    
    return returnCode;
  }

  void EventProcessor::readFile() {
    FDEBUG(1) << " \treadFile\n";
    size_t size = preg_->size();
    SendSourceTerminationSignalIfException sentry(actReg_.get());

    fb_ = input_->readFile();
    if(size < preg_->size()) {
      principalCache_.adjustIndexesAfterProductRegistryAddition();
    }
    principalCache_.adjustEventsToNewProductRegistry(preg());
    if(preallocations_.numberOfStreams()>1 and
        preallocations_.numberOfThreads()>1) {
      fb_->setNotFastClonable(FileBlock::ParallelProcesses);
    }
    sentry.completedSuccessfully();
  }

  void EventProcessor::closeInputFile(bool cleaningUpAfterException) {
    if (fb_.get() != nullptr) {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->closeFile(fb_.get(), cleaningUpAfterException);
      sentry.completedSuccessfully();
    }
    FDEBUG(1) << "\tcloseInputFile\n";
  }

  void EventProcessor::openOutputFiles() {
    if (fb_.get() != nullptr) {
      schedule_->openOutputFiles(*fb_);
      for_all(subProcesses_, [this](auto& subProcess){ subProcess.openOutputFiles(*fb_); });
    }
    FDEBUG(1) << "\topenOutputFiles\n";
  }

  void EventProcessor::closeOutputFiles() {
    if (fb_.get() != nullptr) {
      schedule_->closeOutputFiles();
      for_all(subProcesses_, [](auto& subProcess){ subProcess.closeOutputFiles(); });
    }
    FDEBUG(1) << "\tcloseOutputFiles\n";
  }

  void EventProcessor::respondToOpenInputFile() {
    for_all(subProcesses_, [this](auto& subProcess){ subProcess.updateBranchIDListHelper(branchIDListHelper_->branchIDLists()); } );
    if (fb_.get() != nullptr) {
      schedule_->respondToOpenInputFile(*fb_);
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.respondToOpenInputFile(*fb_); });
    }
    FDEBUG(1) << "\trespondToOpenInputFile\n";
  }

  void EventProcessor::respondToCloseInputFile() {
    if (fb_.get() != nullptr) {
      schedule_->respondToCloseInputFile(*fb_);
      for_all(subProcesses_, [this](auto& subProcess){ subProcess.respondToCloseInputFile(*fb_); });
    }
    FDEBUG(1) << "\trespondToCloseInputFile\n";
  }

  void EventProcessor::startingNewLoop() {
    shouldWeStop_ = false;
    //NOTE: for first loop, need to delay calling 'doStartingNewLoop'
    // until after we've called beginOfJob
    if(looper_ && looperBeginJobRun_) {
      looper_->doStartingNewLoop();
    }
    FDEBUG(1) << "\tstartingNewLoop\n";
  }

  bool EventProcessor::endOfLoop() {
    if(looper_) {
      ModuleChanger changer(schedule_.get(),preg_.get());
      looper_->setModuleChanger(&changer);
      EDLooperBase::Status status = looper_->doEndOfLoop(esp_->eventSetup());
      looper_->setModuleChanger(nullptr);
      if(status != EDLooperBase::kContinue || forceLooperToEnd_) return true;
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

  bool EventProcessor::shouldWeCloseOutput() const {
    FDEBUG(1) << "\tshouldWeCloseOutput\n";
    if(!subProcesses_.empty()) {
      for(auto const& subProcess : subProcesses_) {
        if(subProcess.shouldWeCloseOutput()) {
          return true;
        }
      }
      return false;
    }
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
  }
  
  void EventProcessor::beginRun(ProcessHistoryID const& phid, RunNumber_t run, bool& globalBeginSucceeded) {
    globalBeginSucceeded = false;
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());

      input_->doBeginRun(runPrincipal, &processContext_);
      sentry.completedSuccessfully();
    }

    IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0),
                    runPrincipal.beginTime());
    if(forceESCacheClearOnNewRun_){
      espController_->forceCacheClear();
    }
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      espController_->eventSetupForInstance(ts);
      sentry.completedSuccessfully();
    }
    EventSetup const& es = esp_->eventSetup();
    if(looper_ && looperBeginJobRun_== false) {
      looper_->copyInfo(ScheduleInfo(schedule_.get()));
      looper_->beginOfJob(es);
      looperBeginJobRun_ = true;
      looper_->doStartingNewLoop();
    }
    {
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Traits;
      auto globalWaitTask = make_empty_waiting_task();
      globalWaitTask->increment_ref_count();
      beginGlobalTransitionAsync<Traits>(WaitingTaskHolder(globalWaitTask.get()),
                                         *schedule_,
                                         runPrincipal,
                                         ts,
                                         es,
                                         subProcesses_);
      globalWaitTask->wait_for_all();
      if(globalWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(* (globalWaitTask->exceptionPtr()) );
      }
    }
    globalBeginSucceeded = true;
    FDEBUG(1) << "\tbeginRun " << run << "\n";
    if(looper_) {
      looper_->doBeginRun(runPrincipal, es, &processContext_);
    }
    {
      //To wait, the ref count has to be 1+#streams
      auto streamLoopWaitTask = make_empty_waiting_task();
      streamLoopWaitTask->increment_ref_count();
      
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Traits;
      
      beginStreamsTransitionAsync<Traits>(streamLoopWaitTask.get(),
                                         *schedule_,
                                         preallocations_.numberOfStreams(),
                                         runPrincipal,
                                         ts,
                                         es,
                                         subProcesses_);

      streamLoopWaitTask->wait_for_all();
      if(streamLoopWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(* (streamLoopWaitTask->exceptionPtr()) );
      }
    }
    FDEBUG(1) << "\tstreamBeginRun " << run << "\n";
    if(looper_) {
      //looper_->doStreamBeginRun(schedule_->streamID(),runPrincipal, es);
    }
  }

  void EventProcessor::endRun(ProcessHistoryID const& phid, RunNumber_t run, bool globalBeginSucceeded, bool cleaningUpAfterException) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
    runPrincipal.setEndTime(input_->timestamp());

    IOVSyncValue ts(EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
                    runPrincipal.endTime());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      espController_->eventSetupForInstance(ts);
      sentry.completedSuccessfully();
    }
    EventSetup const& es = esp_->eventSetup();
    if(globalBeginSucceeded){
      //To wait, the ref count has to be 1+#streams
      auto streamLoopWaitTask = make_empty_waiting_task();
      streamLoopWaitTask->increment_ref_count();
      
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Traits;
      
      endStreamsTransitionAsync<Traits>(streamLoopWaitTask.get(),
                                       *schedule_,
                                       preallocations_.numberOfStreams(),
                                       runPrincipal,
                                       ts,
                                       es,
                                       subProcesses_,
                                       cleaningUpAfterException);
      
      streamLoopWaitTask->wait_for_all();
      if(streamLoopWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(* (streamLoopWaitTask->exceptionPtr()) );
      }
    }
    FDEBUG(1) << "\tstreamEndRun " << run << "\n";
    if(looper_) {
      //looper_->doStreamEndRun(schedule_->streamID(),runPrincipal, es);
    }
    {
      auto globalWaitTask = make_empty_waiting_task();
      globalWaitTask->increment_ref_count();

      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Traits;
      endGlobalTransitionAsync<Traits>(WaitingTaskHolder(globalWaitTask.get()),
                                       *schedule_,
                                       runPrincipal,
                                       ts,
                                       es,
                                       subProcesses_,
                                       cleaningUpAfterException);
      globalWaitTask->wait_for_all();
      if(globalWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(* (globalWaitTask->exceptionPtr()) );
      }
    }
    FDEBUG(1) << "\tendRun " << run << "\n";
    if(looper_) {
      looper_->doEndRun(runPrincipal, es, &processContext_);
    }
  }
 
  void
  EventProcessor::beginLumi(std::shared_ptr<LuminosityBlockProcessingStatus>& oStatus) {
    auto waitTask = make_empty_waiting_task();
    waitTask->increment_ref_count();
    
    beginLumiAsync(WaitingTaskHolder{waitTask.get()},oStatus);
    waitTask->wait_for_all();
    
    if(waitTask->exceptionPtr() != nullptr) {
      std::rethrow_exception(* (waitTask->exceptionPtr()) );
    }
  }
  
  void
  EventProcessor::beginLumiAsync(edm::WaitingTaskHolder iHolder, std::shared_ptr<LuminosityBlockProcessingStatus>& oStatus) {
    oStatus= std::make_shared<LuminosityBlockProcessingStatus>(this, preallocations_.numberOfStreams()) ;

    sourceResourcesAcquirer_.serialQueueChain().push([this,iHolder,oStatus]() mutable {
      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      try {
        readLuminosityBlock(*oStatus);

        LuminosityBlockPrincipal& lumiPrincipal = *oStatus->lumiPrincipal();
        {
          SendSourceTerminationSignalIfException sentry(actReg_.get());

          input_->doBeginLumi(lumiPrincipal, &processContext_);
          sentry.completedSuccessfully();
        }

        Service<RandomNumberGenerator> rng;
        if(rng.isAvailable()) {
          LuminosityBlock lb(lumiPrincipal, ModuleDescription(), nullptr, false);
          rng->preBeginLumi(lb);
        }
        
        IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), 0), lumiPrincipal.beginTime());

        //Task to start the stream beginLumis
        auto beginStreamsTask= make_waiting_task(tbb::task::allocate_root()
                                                 ,[this, holder = iHolder, oStatus, ts] (std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            holder.doneWaiting(*iPtr);
          } else {
            oStatus->globalBeginDidSucceed();
            EventSetup const& es = esp_->eventSetup();
            if(looper_) {
              try {
                looper_->doBeginLuminosityBlock(*(oStatus->lumiPrincipal()), es, &processContext_);
              }catch(...) {
                holder.doneWaiting(std::current_exception());
                return;
              }
            }
            typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Traits;
            
            for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
              streamQueues_[i].push([this,i,oStatus,holder,ts,&es] () {
                streamQueues_[i].pause();
                auto& event = principalCache_.eventPrincipal(i);
                streamLumiStatus_[i] = oStatus;
                auto lp = oStatus->lumiPrincipal();
                event.setLuminosityBlockPrincipal(lp);
                beginStreamTransitionAsync<Traits>(holder, *schedule_,i,*lp,ts,es,subProcesses_);
              });
            }
          }
        });
        
        //task to start the global begin lumi
        WaitingTaskHolder beginStreamsHolder{beginStreamsTask};
        auto lumiAction =[this,beginStreamsHolder , oStatus, ts](LimitedTaskQueue::Resumer iResumer) mutable {
          oStatus->setResumer(std::move(iResumer));
          EventSetup const& es = esp_->eventSetup();
          {
            typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
            beginGlobalTransitionAsync<Traits>(beginStreamsHolder,
                                               *schedule_,
                                               *(oStatus->lumiPrincipal()),
                                               ts,
                                               es,
                                               subProcesses_);
          }
        };

        
        //Safe to do check now since can not have multiple beginLumis at same time in this part of the code
        // because we do not attempt to read from the source again until we try to get the first event in a lumi
        if(espController_->isWithinValidityInterval(ts)) {
          iovQueue_.pause();
          lumiQueue_->pushAndPause(std::move(lumiAction));
        } else {
          //If EventSetup fails, need beginStreamsHolder in order to pass back exception
          iovQueue_.push([this,lumiAction,beginStreamsHolder,ts]() mutable {
            try {
              SendSourceTerminationSignalIfException sentry(actReg_.get());
              espController_->eventSetupForInstance(ts);
              sentry.completedSuccessfully();
            } catch(...) {
              beginStreamsHolder.doneWaiting(std::current_exception());
              return;
            }
            iovQueue_.pause();
            lumiQueue_->pushAndPause(std::move(lumiAction));
          });
        }
      } catch(...) {
        iHolder.doneWaiting(std::current_exception());
      }
    });
  }

  void EventProcessor::globalEndLumiAsync(edm::WaitingTaskHolder iTask, std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus) {
    auto t = edm::make_waiting_task(tbb::task::allocate_root(), [t = std::move(iTask), status = iLumiStatus, this] (std::exception_ptr const* iPtr) mutable {
      std::exception_ptr ptr;
      if(iPtr) {
        ptr = *iPtr;
      } else {
        if(looper_) {
          auto& lp = *(status->lumiPrincipal());
          EventSetup const& es = esp_->eventSetup();
          looper_->doEndLuminosityBlock(lp, es, &processContext_);
        }
      }
      t.doneWaiting(ptr);
      //release our hold on the IOV
      iovQueue_.resume();
      status->resumeGlobalLumiQueue();
    });

    auto& lp = *(iLumiStatus->lumiPrincipal());

    IOVSyncValue ts(EventID(lp.run(), lp.luminosityBlock(), EventID::maxEventNumber()),
                    lp.beginTime());


    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;
    EventSetup const& es = esp_->eventSetup();

    endGlobalTransitionAsync<Traits>(WaitingTaskHolder(t),
                                     *schedule_,
                                     lp,
                                     ts,
                                     es,
                                     subProcesses_,
                                     iLumiStatus->cleaningUpAfterException());
  }

  void EventProcessor::streamEndLumiAsync(edm::WaitingTaskHolder iTask,
                                          unsigned int iStreamIndex,
                                          std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus) {
    
    auto t =edm::make_waiting_task(tbb::task::allocate_root(), [this, iStreamIndex, iTask](std::exception_ptr const* iPtr) mutable {
      std::exception_ptr ptr;
      if(iPtr) {
        ptr = *iPtr;
      }
      auto status =streamLumiStatus_[iStreamIndex];
      //reset status before releasing queue else get race condtion
      streamLumiStatus_[iStreamIndex].reset();
      streamQueues_[iStreamIndex].resume();
      
      //are we the last one?
      if( status->streamFinishedLumi()) {
        globalEndLumiAsync(iTask, std::move(status));
      }
      iTask.doneWaiting(ptr);
    });
    
    edm::WaitingTaskHolder lumiDoneTask{t};
    
    iLumiStatus->setEndTime();
    
    if(iLumiStatus->didGlobalBeginSucceed()) {
      auto & lumiPrincipal = *iLumiStatus->lumiPrincipal();
      IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), EventID::maxEventNumber()),
                      lumiPrincipal.endTime());
      EventSetup const& es = esp_->eventSetup();

      bool cleaningUpAfterException = iLumiStatus->cleaningUpAfterException();
      
      using Traits =  OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;
      endStreamTransitionAsync<Traits>(std::move(lumiDoneTask),
                                       *schedule_,iStreamIndex,
                                       lumiPrincipal,ts,es,
                                       subProcesses_,cleaningUpAfterException);
    }
  }

  
  void EventProcessor::endLumi(std::shared_ptr<LuminosityBlockProcessingStatus> status) {
    auto globalWaitTask = make_empty_waiting_task();
    globalWaitTask->increment_ref_count();
    {
      WaitingTaskHolder globalTaskHolder{globalWaitTask.get()};
      for(unsigned int i=0; i< preallocations_.numberOfStreams(); ++i) {
        streamEndLumiAsync(globalTaskHolder, i, status);
      }
    }
    globalWaitTask->wait_for_all();
    if(globalWaitTask->exceptionPtr() != nullptr) {
      std::rethrow_exception(* (globalWaitTask->exceptionPtr()) );
    }
  }

  std::pair<ProcessHistoryID,RunNumber_t> EventProcessor::readRun() {
    if (principalCache_.hasRunPrincipal()) {
      throw edm::Exception(edm::errors::LogicError)
        << "EventProcessor::readRun\n"
        << "Illegal attempt to insert run into cache\n"
        << "Contact a Framework Developer\n";
    }
    auto rp = std::make_shared<RunPrincipal>(input_->runAuxiliary(), preg(), *processConfiguration_, historyAppender_.get(), 0);
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readRun(*rp, *historyAppender_);
      sentry.completedSuccessfully();
    }
    assert(input_->reducedProcessHistoryID() == rp->reducedProcessHistoryID());
    principalCache_.insert(rp);
    return std::make_pair(rp->reducedProcessHistoryID(), input_->run());
  }

  std::pair<ProcessHistoryID,RunNumber_t> EventProcessor::readAndMergeRun() {
    principalCache_.merge(input_->runAuxiliary(), preg());
    auto runPrincipal =principalCache_.runPrincipalPtr();
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readAndMergeRun(*runPrincipal);
      sentry.completedSuccessfully();
    }
    assert(input_->reducedProcessHistoryID() == runPrincipal->reducedProcessHistoryID());
    return std::make_pair(runPrincipal->reducedProcessHistoryID(), input_->run());
  }

  void EventProcessor::readLuminosityBlock(LuminosityBlockProcessingStatus& iStatus) {
    if (!principalCache_.hasRunPrincipal()) {
      throw edm::Exception(edm::errors::LogicError)
        << "EventProcessor::readLuminosityBlock\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "Run is invalid\n"
        << "Contact a Framework Developer\n";
    }
    auto lbp = principalCache_.getAvailableLumiPrincipalPtr();
    lbp->setAux(*input_->luminosityBlockAuxiliary());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readLuminosityBlock(*lbp, *historyAppender_);
      sentry.completedSuccessfully();
    }
    lbp->setRunPrincipal(principalCache_.runPrincipalPtr());
    iStatus.lumiPrincipal() = std::move(lbp);
  }

  int EventProcessor::readAndMergeLumi(LuminosityBlockProcessingStatus& iStatus) {
    auto& lumiPrincipal = *iStatus.lumiPrincipal();
    assert(lumiPrincipal.aux().sameIdentity(*input_->luminosityBlockAuxiliary()) or
           input_->processHistoryRegistry().reducedProcessHistoryID(lumiPrincipal.aux().processHistoryID()) == input_->processHistoryRegistry().reducedProcessHistoryID(input_->luminosityBlockAuxiliary()->processHistoryID()));
    bool lumiOK = lumiPrincipal.adjustToNewProductRegistry(*preg());
    assert(lumiOK);
    lumiPrincipal.mergeAuxiliary(*input_->luminosityBlockAuxiliary());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readAndMergeLumi(*iStatus.lumiPrincipal());
      sentry.completedSuccessfully();
    }
    return input_->luminosityBlock();
  }

  void EventProcessor::writeRun(ProcessHistoryID const& phid, RunNumber_t run) {
    schedule_->writeRun(principalCache_.runPrincipal(phid, run), &processContext_);
    for_all(subProcesses_, [run,phid](auto& subProcess){ subProcess.writeRun(phid, run); });
    FDEBUG(1) << "\twriteRun " << run << "\n";
  }

  void EventProcessor::deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run) {
    principalCache_.deleteRun(phid, run);
    for_all(subProcesses_, [run,phid](auto& subProcess){ subProcess.deleteRunFromCache(phid, run); });
    FDEBUG(1) << "\tdeleteRunFromCache " << run << "\n";
  }

  void EventProcessor::writeLumi(LuminosityBlockProcessingStatus& iStatus) {
    schedule_->writeLumi(*iStatus.lumiPrincipal(), &processContext_);
    for_all(subProcesses_, [&iStatus](auto& subProcess){ subProcess.writeLumi(*iStatus.lumiPrincipal()); });
    //FDEBUG(1) << "\twriteLumi " << run << "/" << lumi << "\n";
  }

  void EventProcessor::deleteLumiFromCache(LuminosityBlockProcessingStatus& iStatus) {
    for(auto& s: subProcesses_) { s.deleteLumiFromCache(*iStatus.lumiPrincipal());}
    iStatus.lumiPrincipal()->clearPrincipal();
    //FDEBUG(1) << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  bool EventProcessor::readNextEventForStream(unsigned int iStreamIndex,
                                              std::atomic<bool>* finishedProcessingEvents) {
    if(shouldWeStop()) {
      return false;
    }
    
    if(deferredExceptionPtrIsSet_.load(std::memory_order_acquire)) {
      return false;
    }
    
    if(finishedProcessingEvents->load(std::memory_order_acquire)) {
      return false;
    }

    ServiceRegistry::Operate operate(serviceToken_);
    try {
      //need to use lock in addition to the serial task queue because
      // of delayed provenance reading and reading data in response to
      // edm::Refs etc
      std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));
      if(not firstEventInBlock_) {
        //The state machine already called input_->nextItemType
        // and found an event. We can't call input_->nextItemType
        // again since it would move to the next transition
        InputSource::ItemType itemType = input_->nextItemType();
        if (InputSource::IsEvent !=itemType) {
          nextItemTypeFromProcessingEvents_ = itemType;
          finishedProcessingEvents->store(true,std::memory_order_release);
          //std::cerr<<"next item type "<<itemType<<"\n";
          return false;
        }
        if((asyncStopRequestedWhileProcessingEvents_=checkForAsyncStopRequest(asyncStopStatusCodeFromProcessingEvents_))) {
          //std::cerr<<"task told to async stop\n";
          actReg_->preSourceEarlyTerminationSignal_(TerminationOrigin::ExternalSignal);
          return false;
        }
      } else {
        firstEventInBlock_ = false;
      }
      readEvent(iStreamIndex);
    } catch (...) {
      bool expected =false;
      if(deferredExceptionPtrIsSet_.compare_exchange_strong(expected,true)) {
        deferredExceptionPtr_ = std::current_exception();

      }
      return false;
    }
    return true;
  }
  
  void EventProcessor::handleNextEventForStreamAsync(WaitingTaskHolder iTask,
                                                     unsigned int iStreamIndex,
                                                     std::atomic<bool>* finishedProcessingEvents)
  {
    sourceResourcesAcquirer_.serialQueueChain().push([this,finishedProcessingEvents,iTask,iStreamIndex]() mutable {
           ServiceRegistry::Operate operate(serviceToken_);

           try {
             if(readNextEventForStream(iStreamIndex, finishedProcessingEvents) ) {
               auto recursionTask = make_waiting_task(tbb::task::allocate_root(), [this,iTask,iStreamIndex,finishedProcessingEvents](std::exception_ptr const* iPtr) mutable {
                 if(iPtr) {
                   bool expected = false;
                   if(deferredExceptionPtrIsSet_.compare_exchange_strong(expected,true)) {
                     deferredExceptionPtr_ = *iPtr;
                     iTask.doneWaiting(*iPtr);
                   }
                   //the stream will stop now
                   return;
                 }
                 handleNextEventForStreamAsync(iTask, iStreamIndex,finishedProcessingEvents);
               });

               processEventAsync( WaitingTaskHolder(recursionTask), iStreamIndex);
             } else {
               //the stream will stop now
               iTask.doneWaiting(std::exception_ptr{});
             }
           } catch(...) {
             bool expected = false;
             if(deferredExceptionPtrIsSet_.compare_exchange_strong(expected,true)) {
               auto e =std::current_exception();
               deferredExceptionPtr_ = e;
               iTask.doneWaiting(e);
             }
           }
    });
  }

  InputSource::ItemType EventProcessor::readAndProcessEvents() {
    nextItemTypeFromProcessingEvents_ = InputSource::IsEvent; //needed for looper
    asyncStopRequestedWhileProcessingEvents_ = false;

    std::atomic<bool> finishedProcessingEvents{false};
    auto finishedProcessingEventsPtr = &finishedProcessingEvents;

    //The state machine already found the event so
    // we have to avoid looking again
    firstEventInBlock_ = true;

    //To wait, the ref count has to b 1+#streams
    auto eventLoopWaitTask = make_empty_waiting_task();
    eventLoopWaitTask->increment_ref_count();

    const unsigned int kNumStreams = preallocations_.numberOfStreams();
    unsigned int iStreamIndex = 0;
    for(; iStreamIndex<kNumStreams-1; ++iStreamIndex) {
      tbb::task::enqueue( *make_waiting_task(tbb::task::allocate_root(),
                                             [this,iStreamIndex,finishedProcessingEventsPtr,h=WaitingTaskHolder{eventLoopWaitTask.get()}](std::exception_ptr const*){
        handleNextEventForStreamAsync(h,iStreamIndex,finishedProcessingEventsPtr);
      }) );
    }
    //need a temporary Task so that the temporary WaitingTaskHolder assigned to h will go out of scope
    // before the call to spawn_and_wait_for_all
    auto t = make_waiting_task(tbb::task::allocate_root(),
                               [this,iStreamIndex,finishedProcessingEventsPtr,h=WaitingTaskHolder{eventLoopWaitTask.get()}](std::exception_ptr const*){
      handleNextEventForStreamAsync(h,iStreamIndex,finishedProcessingEventsPtr);
    });
    eventLoopWaitTask->spawn_and_wait_for_all( *t);

    //One of the processing threads saw an exception
    if(deferredExceptionPtrIsSet_) {
      std::rethrow_exception(deferredExceptionPtr_);
    }
    return nextItemTypeFromProcessingEvents_.load();
  }

  void EventProcessor::readEvent(unsigned int iStreamIndex) {
    //TODO this will have to become per stream
    auto& event = principalCache_.eventPrincipal(iStreamIndex);
    StreamContext streamContext(event.streamID(), &processContext_);

    SendSourceTerminationSignalIfException sentry(actReg_.get());
    input_->readEvent(event, streamContext);
    
    streamLumiStatus_[iStreamIndex]->updateLastTimestamp(input_->timestamp());
    sentry.completedSuccessfully();

    FDEBUG(1) << "\treadEvent\n";
  }

  void EventProcessor::processEventAsync(WaitingTaskHolder iHolder,
                                         unsigned int iStreamIndex) {
    tbb::task::spawn( *make_functor_task( tbb::task::allocate_root(), [=]() {
      processEventAsyncImpl(iHolder, iStreamIndex);
    }) );
  }
  
  void EventProcessor::processEventAsyncImpl(WaitingTaskHolder iHolder,
                                         unsigned int iStreamIndex) {
    auto pep = &(principalCache_.eventPrincipal(iStreamIndex));

    ServiceRegistry::Operate operate(serviceToken_);
    Service<RandomNumberGenerator> rng;
    if(rng.isAvailable()) {
      Event ev(*pep, ModuleDescription(), nullptr);
      rng->postEventRead(ev);
    }
    
    WaitingTaskHolder finalizeEventTask( make_waiting_task(
                    tbb::task::allocate_root(),
                    [this,pep,iHolder](std::exception_ptr const* iPtr) mutable
             {
               ServiceRegistry::Operate operate(serviceToken_);

               //NOTE: If we have a looper we only have one Stream
               if(looper_) {
                 processEventWithLooper(*pep);
               }
               
               FDEBUG(1) << "\tprocessEvent\n";
               pep->clearEventPrincipal();
               if(iPtr) {
                 iHolder.doneWaiting(*iPtr);
               } else {
                 iHolder.doneWaiting(std::exception_ptr());
               }
             }
                                                           )
                                        );
    WaitingTaskHolder afterProcessTask;
    if(subProcesses_.empty()) {
      afterProcessTask = std::move(finalizeEventTask);
    } else {
      //Need to run SubProcesses after schedule has finished
      // with the event
      afterProcessTask = WaitingTaskHolder(
                   make_waiting_task(tbb::task::allocate_root(),
                                     [this,pep,finalizeEventTask] (std::exception_ptr const* iPtr) mutable
      {
         if(not iPtr) {
           ServiceRegistry::Operate operate(serviceToken_);

           //when run with 1 thread, we want to the order to be what
           // it was before. This requires reversing the order since
           // tasks are run last one in first one out
           for(auto& subProcess: boost::adaptors::reverse(subProcesses_)) {
             subProcess.doEventAsync(finalizeEventTask,*pep);
           }
         } else {
           finalizeEventTask.doneWaiting(*iPtr);
         }
       })
                                           );
    }
    
    schedule_->processOneEventAsync(std::move(afterProcessTask),
                                    iStreamIndex,*pep, esp_->eventSetup());

  }

  void EventProcessor::processEventWithLooper(EventPrincipal& iPrincipal) {
    bool randomAccess = input_->randomAccess();
    ProcessingController::ForwardState forwardState = input_->forwardState();
    ProcessingController::ReverseState reverseState = input_->reverseState();
    ProcessingController pc(forwardState, reverseState, randomAccess);
    
    EDLooperBase::Status status = EDLooperBase::kContinue;
    do {
      
      StreamContext streamContext(iPrincipal.streamID(), &processContext_);
      status = looper_->doDuringLoop(iPrincipal, esp_->eventSetup(), pc, &streamContext);
      
      bool succeeded = true;
      if(randomAccess) {
        if(pc.requestedTransition() == ProcessingController::kToPreviousEvent) {
          input_->skipEvents(-2);
        }
        else if(pc.requestedTransition() == ProcessingController::kToSpecifiedEvent) {
          succeeded = input_->goToEvent(pc.specifiedEventTransition());
        }
      }
      pc.setLastOperationSucceeded(succeeded);
    } while(!pc.lastOperationSucceeded());
    if(status != EDLooperBase::kContinue) shouldWeStop_ = true;
  }

  bool EventProcessor::shouldWeStop() const {
    FDEBUG(1) << "\tshouldWeStop\n";
    if(shouldWeStop_) return true;
    if(!subProcesses_.empty()) {
      for(auto const& subProcess : subProcesses_) {
        if(subProcess.terminate()) {
          return true;
        }
      }
      return false;
    }
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

  bool EventProcessor::setDeferredException(std::exception_ptr iException) {
    bool expected =false;
    if(deferredExceptionPtrIsSet_.compare_exchange_strong(expected,true)) {
      deferredExceptionPtr_ = iException;
      return true;
    }
    return false;
  }

}
