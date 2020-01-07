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
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
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
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "FWCore/Concurrency/interface/WaitingTask.h"
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
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "MessageForSource.h"
#include "MessageForParent.h"
#include "LuminosityBlockProcessingStatus.h"

#include "boost/range/adaptor/reversed.hpp"

#include <cassert>
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
    SendSourceTerminationSignalIfException(edm::ActivityRegistry* iReg) : reg_(iReg) {}
    ~SendSourceTerminationSignalIfException() {
      if (reg_) {
        reg_->preSourceEarlyTerminationSignal_(edm::TerminationOrigin::ExceptionFromThisContext);
      }
    }
    void completedSuccessfully() { reg_ = nullptr; }

  private:
    edm::ActivityRegistry* reg_;  // We do not use propagate_const because the registry itself is mutable.
  };

}  // namespace

namespace edm {

  // ---------------------------------------------------------------
  std::unique_ptr<InputSource> makeInput(ParameterSet& params,
                                         CommonParams const& common,
                                         std::shared_ptr<ProductRegistry> preg,
                                         std::shared_ptr<BranchIDListHelper> branchIDListHelper,
                                         std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
                                         std::shared_ptr<ActivityRegistry> areg,
                                         std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                         PreallocationConfiguration const& allocations) {
    ParameterSet* main_input = params.getPSetForUpdate("@main_input");
    if (main_input == nullptr) {
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
      convertException::wrap([&]() { descriptions.validate(*main_input, std::string("source")); });
    } catch (cms::Exception& iException) {
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

    InputSourceDescription isdesc(md,
                                  preg,
                                  branchIDListHelper,
                                  thinnedAssociationsHelper,
                                  areg,
                                  common.maxEventsInput_,
                                  common.maxLumisInput_,
                                  common.maxSecondsUntilRampdown_,
                                  allocations);

    areg->preSourceConstructionSignal_(md);
    std::unique_ptr<InputSource> input;
    try {
      //even if we have an exception, send the signal
      std::shared_ptr<int> sentry(nullptr, [areg, &md](void*) { areg->postSourceConstructionSignal_(md); });
      convertException::wrap([&]() {
        input = std::unique_ptr<InputSource>(InputSourceFactory::get()->makeInputSource(*main_input, isdesc).release());
        input->preEventReadFromSourceSignal_.connect(std::cref(areg->preEventReadFromSourceSignal_));
        input->postEventReadFromSourceSignal_.connect(std::cref(areg->postEventReadFromSourceSignal_));
      });
    } catch (cms::Exception& iException) {
      std::ostringstream ost;
      ost << "Constructing input source of type " << modtype;
      iException.addContext(ost.str());
      throw;
    }
    return input;
  }

  // ---------------------------------------------------------------
  std::shared_ptr<EDLooperBase> fillLooper(eventsetup::EventSetupsController& esController,
                                           eventsetup::EventSetupProvider& cp,
                                           ParameterSet& params) {
    std::shared_ptr<EDLooperBase> vLooper;

    std::vector<std::string> loopers = params.getParameter<std::vector<std::string>>("@all_loopers");

    if (loopers.empty()) {
      return vLooper;
    }

    assert(1 == loopers.size());

    for (std::vector<std::string>::iterator itName = loopers.begin(), itNameEnd = loopers.end(); itName != itNameEnd;
         ++itName) {
      ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      vLooper = eventsetup::LooperFactory::get()->addTo(esController, cp, *providerPSet);
    }
    return vLooper;
  }

  // ---------------------------------------------------------------
  EventProcessor::EventProcessor(std::unique_ptr<ParameterSet> parameterSet,  //std::string const& config,
                                 ServiceToken const& iToken,
                                 serviceregistry::ServiceLegacy iLegacy,
                                 std::vector<std::string> const& defaultServices,
                                 std::vector<std::string> const& forcedServices)
      : actReg_(),
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
        exceptionMessageLumis_(false),
        forceLooperToEnd_(false),
        looperBeginJobRun_(false),
        forceESCacheClearOnNewRun_(false),
        eventSetupDataToExcludeFromPrefetching_() {
    auto processDesc = std::make_shared<ProcessDesc>(std::move(parameterSet));
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, iToken, iLegacy);
  }

  EventProcessor::EventProcessor(std::unique_ptr<ParameterSet> parameterSet,  //std::string const& config,
                                 std::vector<std::string> const& defaultServices,
                                 std::vector<std::string> const& forcedServices)
      : actReg_(),
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
        exceptionMessageLumis_(false),
        forceLooperToEnd_(false),
        looperBeginJobRun_(false),
        forceESCacheClearOnNewRun_(false),
        asyncStopRequestedWhileProcessingEvents_(false),
        eventSetupDataToExcludeFromPrefetching_() {
    auto processDesc = std::make_shared<ProcessDesc>(std::move(parameterSet));
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
  }

  EventProcessor::EventProcessor(std::shared_ptr<ProcessDesc> processDesc,
                                 ServiceToken const& token,
                                 serviceregistry::ServiceLegacy legacy)
      : actReg_(),
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
        exceptionMessageLumis_(false),
        forceLooperToEnd_(false),
        looperBeginJobRun_(false),
        forceESCacheClearOnNewRun_(false),
        asyncStopRequestedWhileProcessingEvents_(false),
        eventSetupDataToExcludeFromPrefetching_() {
    init(processDesc, token, legacy);
  }

  void EventProcessor::init(std::shared_ptr<ProcessDesc>& processDesc,
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

    // Validates the parameters in the 'options', 'maxEvents', 'maxLuminosityBlocks',
    // and 'maxSecondsUntilRampdown' top level parameter sets. Default values are also
    // set in here if the parameters were not explicitly set.
    validateTopLevelParameterSets(parameterSet.get());

    // Now set some parameters specific to the main process.
    ParameterSet const& optionsPset(parameterSet->getUntrackedParameterSet("options"));
    auto const& fileMode = optionsPset.getUntrackedParameter<std::string>("fileMode");
    if (fileMode != "NOMERGE" and fileMode != "FULLMERGE") {
      throw Exception(errors::Configuration, "Illegal fileMode parameter value: ")
          << fileMode << ".\n"
          << "Legal values are 'NOMERGE' and 'FULLMERGE'.\n";
    } else {
      fileModeNoMerge_ = (fileMode == "NOMERGE");
    }
    forceESCacheClearOnNewRun_ = optionsPset.getUntrackedParameter<bool>("forceEventSetupCacheClearOnNewRun");

    //threading
    unsigned int nThreads = optionsPset.getUntrackedParameter<unsigned int>("numberOfThreads");

    // Even if numberOfThreads was set to zero in the Python configuration, the code
    // in cmsRun.cpp should have reset it to something else.
    assert(nThreads != 0);

    unsigned int nStreams = optionsPset.getUntrackedParameter<unsigned int>("numberOfStreams");
    if (nStreams == 0) {
      nStreams = nThreads;
    }
    if (nThreads > 1 or nStreams > 1) {
      edm::LogInfo("ThreadStreamSetup") << "setting # threads " << nThreads << "\nsetting # streams " << nStreams;
    }
    unsigned int nConcurrentRuns = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentRuns");
    if (nConcurrentRuns != 1) {
      throw Exception(errors::Configuration, "Illegal value nConcurrentRuns : ")
          << "Although the plan is to change this in the future, currently nConcurrentRuns must always be 1.\n";
    }
    unsigned int nConcurrentLumis =
        optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentLuminosityBlocks");
    if (nConcurrentLumis == 0) {
      nConcurrentLumis = nConcurrentRuns;
    }

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
    IllegalParameters::setThrowAnException(optionsPset.getUntrackedParameter<bool>("throwIfIllegalParameter"));

    printDependencies_ = optionsPset.getUntrackedParameter<bool>("printDependencies");

    // Now do general initialization
    ScheduleItems items;

    //initialize the services
    auto& serviceSets = processDesc->getServicesPSets();
    ServiceToken token = items.initServices(serviceSets, *parameterSet, iToken, iLegacy, true);
    serviceToken_ = items.addCPRandTNS(*parameterSet, token);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    if (nStreams > 1) {
      edm::Service<RootHandlers> handler;
      handler->willBeUsingThreads();
    }

    // intialize miscellaneous items
    std::shared_ptr<CommonParams> common(items.initMisc(*parameterSet));

    // intialize the event setup provider
    ParameterSet const& eventSetupPset(optionsPset.getUntrackedParameterSet("eventSetup"));
    esp_ = espController_->makeProvider(*parameterSet, items.actReg_.get(), &eventSetupPset);

    // initialize the looper, if any
    looper_ = fillLooper(*espController_, *esp_, *parameterSet);
    if (looper_) {
      looper_->setActionTable(items.act_table_.get());
      looper_->attachTo(*items.actReg_);

      //For now loopers make us run only 1 transition at a time
      nStreams = 1;
      nConcurrentLumis = 1;
      nConcurrentRuns = 1;
    }
    espController_->setMaxConcurrentIOVs(nStreams, nConcurrentLumis);

    preallocations_ = PreallocationConfiguration{nThreads, nStreams, nConcurrentLumis, nConcurrentRuns};

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
    schedule_ = items.initSchedule(*parameterSet, hasSubProcesses, preallocations_, &processContext_);

    // set the data members
    act_table_ = std::move(items.act_table_);
    actReg_ = items.actReg_;
    preg_ = items.preg();
    mergeableRunProductProcesses_.setProcessesWithMergeableRunProducts(*preg_);
    branchIDListHelper_ = items.branchIDListHelper();
    thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
    processConfiguration_ = items.processConfiguration();
    processContext_.setProcessConfiguration(processConfiguration_.get());
    principalCache_.setProcessHistoryRegistry(input_->processHistoryRegistry());

    FDEBUG(2) << parameterSet << std::endl;

    principalCache_.setNumberOfConcurrentPrincipals(preallocations_);
    for (unsigned int index = 0; index < preallocations_.numberOfStreams(); ++index) {
      // Reusable event principal
      auto ep = std::make_shared<EventPrincipal>(preg(),
                                                 branchIDListHelper(),
                                                 thinnedAssociationsHelper(),
                                                 *processConfiguration_,
                                                 historyAppender_.get(),
                                                 index);
      principalCache_.insert(std::move(ep));
    }

    for (unsigned int index = 0; index < preallocations_.numberOfLuminosityBlocks(); ++index) {
      auto lp =
          std::make_unique<LuminosityBlockPrincipal>(preg(), *processConfiguration_, historyAppender_.get(), index);
      principalCache_.insert(std::move(lp));
    }

    // fill the subprocesses, if there are any
    subProcesses_.reserve(subProcessVParameterSet.size());
    for (auto& subProcessPSet : subProcessVParameterSet) {
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

  void EventProcessor::beginJob() {
    if (beginJobCalled_)
      return;
    beginJobCalled_ = true;
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

    if (preallocations_.numberOfLuminosityBlocks() > 1) {
      warnAboutModulesRequiringLuminosityBLockSynchronization();
    }
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
      convertException::wrap([&]() { input_->doBeginJob(); });
    } catch (cms::Exception& ex) {
      ex.addContext("Calling beginJob for the source");
      throw;
    }
    espController_->finishConfiguration();
    schedule_->beginJob(*preg_, esp_->recordsToProxyIndices());
    // toerror.succeeded(); // should we add this?
    for_all(subProcesses_, [](auto& subProcess) { subProcess.doBeginJob(); });
    actReg_->postBeginJobSignal_();

    for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
      schedule_->beginStream(i);
      for_all(subProcesses_, [i](auto& subProcess) { subProcess.doBeginStream(i); });
    }
  }

  void EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c(
        "Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    //NOTE: this really should go elsewhere in the future
    for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
      c.call([this, i]() { this->schedule_->endStream(i); });
      for (auto& subProcess : subProcesses_) {
        c.call([&subProcess, i]() { subProcess.doEndStream(i); });
      }
    }
    auto actReg = actReg_.get();
    c.call([actReg]() { actReg->preEndJobSignal_(); });
    schedule_->endJob(c);
    for (auto& subProcess : subProcesses_) {
      c.call(std::bind(&SubProcess::doEndJob, &subProcess));
    }
    c.call(std::bind(&InputSource::doEndJob, input_.get()));
    if (looper_) {
      c.call(std::bind(&EDLooperBase::endOfJob, looper()));
    }
    c.call([actReg]() { actReg->postEndJobSignal_(); });
    if (c.hasThrown()) {
      c.rethrow();
    }
  }

  ServiceToken EventProcessor::getToken() { return serviceToken_; }

  std::vector<ModuleDescription const*> EventProcessor::getAllModuleDescriptions() const {
    return schedule_->getAllModuleDescriptions();
  }

  int EventProcessor::totalEvents() const { return schedule_->totalEvents(); }

  int EventProcessor::totalEventsPassed() const { return schedule_->totalEventsPassed(); }

  int EventProcessor::totalEventsFailed() const { return schedule_->totalEventsFailed(); }

  void EventProcessor::enableEndPaths(bool active) { schedule_->enableEndPaths(active); }

  bool EventProcessor::endPathsEnabled() const { return schedule_->endPathsEnabled(); }

  void EventProcessor::getTriggerReport(TriggerReport& rep) const { schedule_->getTriggerReport(rep); }

  void EventProcessor::clearCounters() { schedule_->clearCounters(); }

  namespace {
#include "TransitionProcessors.icc"
  }

  bool EventProcessor::checkForAsyncStopRequest(StatusCode& returnCode) {
    bool returnValue = false;

    // Look for a shutdown signal
    if (shutdown_flag.load(std::memory_order_acquire)) {
      returnValue = true;
      returnCode = epSignal;
    }
    return returnValue;
  }

  InputSource::ItemType EventProcessor::nextTransitionType() {
    if (deferredExceptionPtrIsSet_.load()) {
      lastSourceTransition_ = InputSource::IsStop;
      return InputSource::IsStop;
    }

    SendSourceTerminationSignalIfException sentry(actReg_.get());
    InputSource::ItemType itemType;
    //For now, do nothing with InputSource::IsSynchronize
    do {
      itemType = input_->nextItemType();
    } while (itemType == InputSource::IsSynchronize);

    lastSourceTransition_ = itemType;
    sentry.completedSuccessfully();

    StatusCode returnCode = epSuccess;

    if (checkForAsyncStopRequest(returnCode)) {
      actReg_->preSourceEarlyTerminationSignal_(TerminationOrigin::ExternalSignal);
      lastSourceTransition_ = InputSource::IsStop;
    }

    return lastSourceTransition_;
  }

  std::pair<edm::ProcessHistoryID, edm::RunNumber_t> EventProcessor::nextRunID() {
    return std::make_pair(input_->reducedProcessHistoryID(), input_->run());
  }

  edm::LuminosityBlockNumber_t EventProcessor::nextLuminosityBlockID() { return input_->luminosityBlock(); }

  EventProcessor::StatusCode EventProcessor::runToCompletion() {
    StatusCode returnCode = epSuccess;
    asyncStopStatusCodeFromProcessingEvents_ = epSuccess;
    {
      beginJob();  //make sure this was called

      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      asyncStopRequestedWhileProcessingEvents_ = false;
      try {
        FilesProcessor fp(fileModeNoMerge_);

        convertException::wrap([&]() {
          bool firstTime = true;
          do {
            if (not firstTime) {
              prepareForNextLoop();
              rewindInput();
            } else {
              firstTime = false;
            }
            startingNewLoop();

            auto trans = fp.processFiles(*this);

            fp.normalEnd();

            if (deferredExceptionPtrIsSet_.load()) {
              std::rethrow_exception(deferredExceptionPtr_);
            }
            if (trans != InputSource::IsStop) {
              //problem with the source
              doErrorStuff();

              throw cms::Exception("BadTransition") << "Unexpected transition change " << trans;
            }
          } while (not endOfLoop());
        });  // convertException::wrap

      }  // Try block
      catch (cms::Exception& e) {
        if (exceptionMessageLumis_) {
          std::string message(
              "Another exception was caught while trying to clean up lumis after the primary fatal exception.");
          e.addAdditionalInfo(message);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << message;
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

    principalCache_.preReadFile();

    fb_ = input_->readFile();
    if (size < preg_->size()) {
      principalCache_.adjustIndexesAfterProductRegistryAddition();
    }
    principalCache_.adjustEventsToNewProductRegistry(preg());
    if (preallocations_.numberOfStreams() > 1 and preallocations_.numberOfThreads() > 1) {
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
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.openOutputFiles(*fb_); });
    }
    FDEBUG(1) << "\topenOutputFiles\n";
  }

  void EventProcessor::closeOutputFiles() {
    if (fb_.get() != nullptr) {
      schedule_->closeOutputFiles();
      for_all(subProcesses_, [](auto& subProcess) { subProcess.closeOutputFiles(); });
    }
    FDEBUG(1) << "\tcloseOutputFiles\n";
  }

  void EventProcessor::respondToOpenInputFile() {
    for_all(subProcesses_,
            [this](auto& subProcess) { subProcess.updateBranchIDListHelper(branchIDListHelper_->branchIDLists()); });
    if (fb_.get() != nullptr) {
      schedule_->respondToOpenInputFile(*fb_);
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.respondToOpenInputFile(*fb_); });
    }
    FDEBUG(1) << "\trespondToOpenInputFile\n";
  }

  void EventProcessor::respondToCloseInputFile() {
    if (fb_.get() != nullptr) {
      schedule_->respondToCloseInputFile(*fb_);
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.respondToCloseInputFile(*fb_); });
    }
    FDEBUG(1) << "\trespondToCloseInputFile\n";
  }

  void EventProcessor::startingNewLoop() {
    shouldWeStop_ = false;
    //NOTE: for first loop, need to delay calling 'doStartingNewLoop'
    // until after we've called beginOfJob
    if (looper_ && looperBeginJobRun_) {
      looper_->doStartingNewLoop();
    }
    FDEBUG(1) << "\tstartingNewLoop\n";
  }

  bool EventProcessor::endOfLoop() {
    if (looper_) {
      ModuleChanger changer(schedule_.get(), preg_.get(), esp_->recordsToProxyIndices());
      looper_->setModuleChanger(&changer);
      EDLooperBase::Status status = looper_->doEndOfLoop(esp_->eventSetupImpl());
      looper_->setModuleChanger(nullptr);
      if (status != EDLooperBase::kContinue || forceLooperToEnd_)
        return true;
      else
        return false;
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
    if (!subProcesses_.empty()) {
      for (auto const& subProcess : subProcesses_) {
        if (subProcess.shouldWeCloseOutput()) {
          return true;
        }
      }
      return false;
    }
    return schedule_->shouldWeCloseOutput();
  }

  void EventProcessor::doErrorStuff() {
    FDEBUG(1) << "\tdoErrorStuff\n";
    LogError("StateMachine") << "The EventProcessor state machine encountered an unexpected event\n"
                             << "and went to the error state\n"
                             << "Will attempt to terminate processing normally\n"
                             << "(IF using the looper the next loop will be attempted)\n"
                             << "This likely indicates a bug in an input module or corrupted input or both\n";
  }

  void EventProcessor::beginRun(ProcessHistoryID const& phid,
                                RunNumber_t run,
                                bool& globalBeginSucceeded,
                                bool& eventSetupForInstanceSucceeded) {
    globalBeginSucceeded = false;
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());

      input_->doBeginRun(runPrincipal, &processContext_);
      sentry.completedSuccessfully();
    }

    IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0), runPrincipal.beginTime());
    if (forceESCacheClearOnNewRun_) {
      espController_->forceCacheClear();
    }
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      espController_->eventSetupForInstance(ts);
      eventSetupForInstanceSucceeded = true;
      sentry.completedSuccessfully();
    }
    auto const& es = esp_->eventSetupImpl();
    if (looper_ && looperBeginJobRun_ == false) {
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
                                         nullptr,
                                         serviceToken_,
                                         subProcesses_);
      globalWaitTask->wait_for_all();
      if (globalWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
      }
    }
    globalBeginSucceeded = true;
    FDEBUG(1) << "\tbeginRun " << run << "\n";
    if (looper_) {
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
                                          nullptr,
                                          serviceToken_,
                                          subProcesses_);

      streamLoopWaitTask->wait_for_all();
      if (streamLoopWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
      }
    }
    FDEBUG(1) << "\tstreamBeginRun " << run << "\n";
    if (looper_) {
      //looper_->doStreamBeginRun(schedule_->streamID(),runPrincipal, es);
    }
  }

  void EventProcessor::endUnfinishedRun(ProcessHistoryID const& phid,
                                        RunNumber_t run,
                                        bool globalBeginSucceeded,
                                        bool cleaningUpAfterException,
                                        bool eventSetupForInstanceSucceeded) {
    if (eventSetupForInstanceSucceeded) {
      //If we skip empty runs, this would be called conditionally
      endRun(phid, run, globalBeginSucceeded, cleaningUpAfterException);

      if (globalBeginSucceeded) {
        auto t = edm::make_empty_waiting_task();
        t->increment_ref_count();
        RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
        MergeableRunProductMetadata* mergeableRunProductMetadata = runPrincipal.mergeableRunProductMetadata();
        mergeableRunProductMetadata->preWriteRun();
        writeRunAsync(edm::WaitingTaskHolder{t.get()}, phid, run, mergeableRunProductMetadata);
        t->wait_for_all();
        mergeableRunProductMetadata->postWriteRun();
        if (t->exceptionPtr()) {
          std::rethrow_exception(*t->exceptionPtr());
        }
      }
    }
    deleteRunFromCache(phid, run);
  }

  void EventProcessor::endRun(ProcessHistoryID const& phid,
                              RunNumber_t run,
                              bool globalBeginSucceeded,
                              bool cleaningUpAfterException) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
    runPrincipal.setEndTime(input_->timestamp());

    IOVSyncValue ts(
        EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
        runPrincipal.endTime());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      espController_->eventSetupForInstance(ts);
      sentry.completedSuccessfully();
    }
    auto const& es = esp_->eventSetupImpl();
    if (globalBeginSucceeded) {
      //To wait, the ref count has to be 1+#streams
      auto streamLoopWaitTask = make_empty_waiting_task();
      streamLoopWaitTask->increment_ref_count();

      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Traits;

      endStreamsTransitionAsync<Traits>(WaitingTaskHolder(streamLoopWaitTask.get()),
                                        *schedule_,
                                        preallocations_.numberOfStreams(),
                                        runPrincipal,
                                        ts,
                                        es,
                                        nullptr,
                                        serviceToken_,
                                        subProcesses_,
                                        cleaningUpAfterException);

      streamLoopWaitTask->wait_for_all();
      if (streamLoopWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
      }
    }
    FDEBUG(1) << "\tstreamEndRun " << run << "\n";
    if (looper_) {
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
                                       nullptr,
                                       serviceToken_,
                                       subProcesses_,
                                       cleaningUpAfterException);
      globalWaitTask->wait_for_all();
      if (globalWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
      }
    }
    FDEBUG(1) << "\tendRun " << run << "\n";
    if (looper_) {
      looper_->doEndRun(runPrincipal, es, &processContext_);
    }
  }

  InputSource::ItemType EventProcessor::processLumis(std::shared_ptr<void> const& iRunResource) {
    auto waitTask = make_empty_waiting_task();
    waitTask->increment_ref_count();

    if (streamLumiActive_ > 0) {
      assert(streamLumiActive_ == preallocations_.numberOfStreams());
      // Continue after opening a new input file
      continueLumiAsync(WaitingTaskHolder{waitTask.get()});
    } else {
      beginLumiAsync(IOVSyncValue(EventID(input_->run(), input_->luminosityBlock(), 0),
                                  input_->luminosityBlockAuxiliary()->beginTime()),
                     iRunResource,
                     WaitingTaskHolder{waitTask.get()});
    }
    waitTask->wait_for_all();

    if (waitTask->exceptionPtr() != nullptr) {
      std::rethrow_exception(*(waitTask->exceptionPtr()));
    }
    return lastTransitionType();
  }

  void EventProcessor::beginLumiAsync(IOVSyncValue const& iSync,
                                      std::shared_ptr<void> const& iRunResource,
                                      edm::WaitingTaskHolder iHolder) {
    if (iHolder.taskHasFailed()) {
      return;
    }

    // We must be careful with the status object here and in code this function calls. IF we want
    // endRun to be called, then we must call resetResources before the things waiting on
    // iHolder are allowed to proceed. Otherwise, there will be race condition (possibly causing
    // endRun to be called much later than it should be, because status is holding iRunResource).
    // Note that this must be done explicitly. Relying on the destructor does not work well
    // because the LimitedTaskQueue for the lumiWork holds the shared_ptr in each of its internal
    // queues, plus it is difficult to guarantee the destructor is called  before iHolder gets
    // destroyed inside this function and lumiWork.
    auto status =
        std::make_shared<LuminosityBlockProcessingStatus>(this, preallocations_.numberOfStreams(), iRunResource);

    auto lumiWork = [this, iHolder, status](edm::LimitedTaskQueue::Resumer iResumer) mutable {
      if (iHolder.taskHasFailed()) {
        status->resetResources();
        return;
      }

      status->setResumer(std::move(iResumer));

      sourceResourcesAcquirer_.serialQueueChain().push([this, iHolder, status = std::move(status)]() mutable {
        //make the services available
        ServiceRegistry::Operate operate(serviceToken_);
        // Caught exception is propagated via WaitingTaskHolder
        CMS_SA_ALLOW try {
          readLuminosityBlock(*status);

          LuminosityBlockPrincipal& lumiPrincipal = *status->lumiPrincipal();
          {
            SendSourceTerminationSignalIfException sentry(actReg_.get());

            input_->doBeginLumi(lumiPrincipal, &processContext_);
            sentry.completedSuccessfully();
          }

          Service<RandomNumberGenerator> rng;
          if (rng.isAvailable()) {
            LuminosityBlock lb(lumiPrincipal, ModuleDescription(), nullptr, false);
            rng->preBeginLumi(lb);
          }

          IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), 0), lumiPrincipal.beginTime());

          //Task to start the stream beginLumis
          auto beginStreamsTask = make_waiting_task(
              tbb::task::allocate_root(), [this, holder = iHolder, status, ts](std::exception_ptr const* iPtr) mutable {
                if (iPtr) {
                  status->resetResources();
                  holder.doneWaiting(*iPtr);
                } else {
                  status->globalBeginDidSucceed();
                  EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());

                  if (looper_) {
                    // Caught exception is propagated via WaitingTaskHolder
                    CMS_SA_ALLOW try {
                      //make the services available
                      ServiceRegistry::Operate operateLooper(serviceToken_);
                      looper_->doBeginLuminosityBlock(*(status->lumiPrincipal()), es, &processContext_);
                    } catch (...) {
                      status->resetResources();
                      holder.doneWaiting(std::current_exception());
                      return;
                    }
                  }
                  typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Traits;

                  for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
                    streamQueues_[i].push([this, i, status, holder, ts, &es]() mutable {
                      streamQueues_[i].pause();

                      auto eventTask = edm::make_waiting_task(
                          tbb::task::allocate_root(),
                          [this, i, h = holder](std::exception_ptr const* exceptionFromBeginStreamLumi) mutable {
                            if (exceptionFromBeginStreamLumi) {
                              WaitingTaskHolder tmp(h);
                              tmp.doneWaiting(*exceptionFromBeginStreamLumi);
                              streamEndLumiAsync(h, i, streamLumiStatus_[i]);
                            } else {
                              handleNextEventForStreamAsync(std::move(h), i);
                            }
                          });
                      auto& event = principalCache_.eventPrincipal(i);
                      streamLumiStatus_[i] = status;
                      ++streamLumiActive_;
                      auto lp = status->lumiPrincipal();
                      event.setLuminosityBlockPrincipal(lp.get());
                      beginStreamTransitionAsync<Traits>(WaitingTaskHolder{eventTask},
                                                         *schedule_,
                                                         i,
                                                         *lp,
                                                         ts,
                                                         es,
                                                         &status->eventSetupImpls(),
                                                         serviceToken_,
                                                         subProcesses_);
                    });
                  }
                }
              });  // beginStreamTask

          //task to start the global begin lumi
          WaitingTaskHolder beginStreamsHolder{beginStreamsTask};

          EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());
          {
            typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
            beginGlobalTransitionAsync<Traits>(beginStreamsHolder,
                                               *schedule_,
                                               lumiPrincipal,
                                               ts,
                                               es,
                                               &status->eventSetupImpls(),
                                               serviceToken_,
                                               subProcesses_);
          }
        } catch (...) {
          status->resetResources();
          iHolder.doneWaiting(std::current_exception());
        }
      });  // task in sourceResourcesAcquirer
    };     // end lumiWork

    auto queueLumiWorkTask = make_waiting_task(
        tbb::task::allocate_root(),
        [this, lumiWorkLambda = std::move(lumiWork), iHolder](std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            iHolder.doneWaiting(*iPtr);
          }
          lumiQueue_->pushAndPause(std::move(lumiWorkLambda));
        });

    if (espController_->doWeNeedToWaitForIOVsToFinish(iSync)) {
      // We only get here inside this block if there is an EventSetup
      // module not able to handle concurrent IOVs (usually an ESSource)
      // and the new sync value is outside the current IOV of that module.

      WaitingTaskHolder queueLumiWorkTaskHolder{queueLumiWorkTask};

      queueWhichWaitsForIOVsToFinish_.push([this, queueLumiWorkTaskHolder, iSync, status]() mutable {
        // Caught exception is propagated via WaitingTaskHolder
        CMS_SA_ALLOW try {
          SendSourceTerminationSignalIfException sentry(actReg_.get());
          // Pass in iSync to let the EventSetup system know which run and lumi
          // need to be processed and prepare IOVs for it.
          // Pass in the endIOVWaitingTasks so the lumi can notify them when the
          // lumi is done and no longer needs its EventSetup IOVs.
          espController_->eventSetupForInstance(
              iSync, queueLumiWorkTaskHolder, status->endIOVWaitingTasks(), status->eventSetupImpls());
          sentry.completedSuccessfully();
        } catch (...) {
          queueLumiWorkTaskHolder.doneWaiting(std::current_exception());
        }
        queueWhichWaitsForIOVsToFinish_.pause();
      });

    } else {
      queueWhichWaitsForIOVsToFinish_.pause();

      // This holder will be used to wait until the EventSetup IOVs are ready
      WaitingTaskHolder queueLumiWorkTaskHolder{queueLumiWorkTask};
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try {
        SendSourceTerminationSignalIfException sentry(actReg_.get());

        // Pass in iSync to let the EventSetup system know which run and lumi
        // need to be processed and prepare IOVs for it.
        // Pass in the endIOVWaitingTasks so the lumi can notify them when the
        // lumi is done and no longer needs its EventSetup IOVs.
        espController_->eventSetupForInstance(
            iSync, queueLumiWorkTaskHolder, status->endIOVWaitingTasks(), status->eventSetupImpls());
        sentry.completedSuccessfully();

      } catch (...) {
        queueLumiWorkTaskHolder.doneWaiting(std::current_exception());
      }
    }
  }

  void EventProcessor::continueLumiAsync(edm::WaitingTaskHolder iHolder) {
    {
      //all streams are sharing the same status at the moment
      auto status = streamLumiStatus_[0];  //read from streamLumiActive_ happened in calling routine
      status->needToContinueLumi();
      status->startProcessingEvents();
    }

    unsigned int streamIndex = 0;
    for (; streamIndex < preallocations_.numberOfStreams() - 1; ++streamIndex) {
      tbb::task::enqueue(*edm::make_functor_task(tbb::task::allocate_root(), [this, streamIndex, h = iHolder]() {
        handleNextEventForStreamAsync(std::move(h), streamIndex);
      }));
    }
    tbb::task::spawn(*edm::make_functor_task(tbb::task::allocate_root(), [this, streamIndex, h = std::move(iHolder)]() {
      handleNextEventForStreamAsync(std::move(h), streamIndex);
    }));
  }

  void EventProcessor::handleEndLumiExceptions(std::exception_ptr const* iPtr, WaitingTaskHolder& holder) {
    if (setDeferredException(*iPtr)) {
      WaitingTaskHolder tmp(holder);
      tmp.doneWaiting(*iPtr);
    } else {
      setExceptionMessageLumis();
    }
  }

  void EventProcessor::globalEndLumiAsync(edm::WaitingTaskHolder iTask,
                                          std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus) {
    // Get some needed info out of the status object before moving
    // it into finalTaskForThisLumi.
    auto& lp = *(iLumiStatus->lumiPrincipal());
    bool didGlobalBeginSucceed = iLumiStatus->didGlobalBeginSucceed();
    bool cleaningUpAfterException = iLumiStatus->cleaningUpAfterException();
    EventSetupImpl const& es = iLumiStatus->eventSetupImpl(esp_->subProcessIndex());
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = &iLumiStatus->eventSetupImpls();

    auto finalTaskForThisLumi = edm::make_waiting_task(
        tbb::task::allocate_root(),
        [status = std::move(iLumiStatus), iTask = std::move(iTask), this](std::exception_ptr const* iPtr) mutable {
          std::exception_ptr ptr;
          if (iPtr) {
            handleEndLumiExceptions(iPtr, iTask);
          } else {
            // Caught exception is passed to handleEndLumiExceptions()
            CMS_SA_ALLOW try {
              ServiceRegistry::Operate operate(serviceToken_);
              if (looper_) {
                auto& lumiPrincipal = *(status->lumiPrincipal());
                EventSetupImpl const& eventSetupImpl = status->eventSetupImpl(esp_->subProcessIndex());
                looper_->doEndLuminosityBlock(lumiPrincipal, eventSetupImpl, &processContext_);
              }
            } catch (...) {
              ptr = std::current_exception();
            }
          }
          ServiceRegistry::Operate operate(serviceToken_);

          // Try hard to clean up resources so the
          // process can terminate in a controlled
          // fashion even after exceptions have occurred.
          // Caught exception is passed to handleEndLumiExceptions()
          CMS_SA_ALLOW try { deleteLumiFromCache(*status); } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }
          // Caught exception is passed to handleEndLumiExceptions()
          CMS_SA_ALLOW try {
            status->resumeGlobalLumiQueue();
            queueWhichWaitsForIOVsToFinish_.resume();
          } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }
          // Caught exception is passed to handleEndLumiExceptions()
          CMS_SA_ALLOW try {
            // This call to status.resetResources() must occur before iTask is destroyed.
            // Otherwise there will be a data race which could result in endRun
            // being delayed until it is too late to successfully call it.
            status->resetResources();
            status.reset();
          } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }

          if (ptr) {
            handleEndLumiExceptions(&ptr, iTask);
          }
        });

    auto writeT = edm::make_waiting_task(
        tbb::task::allocate_root(),
        [this, didGlobalBeginSucceed, &lumiPrincipal = lp, task = WaitingTaskHolder(finalTaskForThisLumi)](
            std::exception_ptr const* iExcept) mutable {
          if (iExcept) {
            task.doneWaiting(*iExcept);
          } else {
            //Only call writeLumi if beginLumi succeeded
            if (didGlobalBeginSucceed) {
              writeLumiAsync(std::move(task), lumiPrincipal);
            }
          }
        });

    IOVSyncValue ts(EventID(lp.run(), lp.luminosityBlock(), EventID::maxEventNumber()), lp.beginTime());

    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;

    endGlobalTransitionAsync<Traits>(WaitingTaskHolder(writeT),
                                     *schedule_,
                                     lp,
                                     ts,
                                     es,
                                     eventSetupImpls,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
  }

  void EventProcessor::streamEndLumiAsync(edm::WaitingTaskHolder iTask,
                                          unsigned int iStreamIndex,
                                          std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus) {
    auto t = edm::make_waiting_task(tbb::task::allocate_root(),
                                    [this, iStreamIndex, iTask](std::exception_ptr const* iPtr) mutable {
                                      if (iPtr) {
                                        handleEndLumiExceptions(iPtr, iTask);
                                      }
                                      auto status = streamLumiStatus_[iStreamIndex];
                                      //reset status before releasing queue else get race condtion
                                      streamLumiStatus_[iStreamIndex].reset();
                                      --streamLumiActive_;
                                      streamQueues_[iStreamIndex].resume();

                                      //are we the last one?
                                      if (status->streamFinishedLumi()) {
                                        globalEndLumiAsync(iTask, std::move(status));
                                      }
                                    });

    edm::WaitingTaskHolder lumiDoneTask{t};

    iLumiStatus->setEndTime();

    if (iLumiStatus->didGlobalBeginSucceed()) {
      auto& lumiPrincipal = *iLumiStatus->lumiPrincipal();
      IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), EventID::maxEventNumber()),
                      lumiPrincipal.endTime());
      EventSetupImpl const& es = iLumiStatus->eventSetupImpl(esp_->subProcessIndex());

      bool cleaningUpAfterException = iLumiStatus->cleaningUpAfterException();

      using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;
      endStreamTransitionAsync<Traits>(std::move(lumiDoneTask),
                                       *schedule_,
                                       iStreamIndex,
                                       lumiPrincipal,
                                       ts,
                                       es,
                                       &iLumiStatus->eventSetupImpls(),
                                       serviceToken_,
                                       subProcesses_,
                                       cleaningUpAfterException);
    }
  }

  void EventProcessor::endUnfinishedLumi() {
    if (streamLumiActive_.load() > 0) {
      auto globalWaitTask = make_empty_waiting_task();
      globalWaitTask->increment_ref_count();
      {
        WaitingTaskHolder globalTaskHolder{globalWaitTask.get()};
        for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
          if (streamLumiStatus_[i]) {
            streamEndLumiAsync(globalTaskHolder, i, streamLumiStatus_[i]);
          }
        }
      }
      globalWaitTask->wait_for_all();
      if (globalWaitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
      }
    }
  }

  std::pair<ProcessHistoryID, RunNumber_t> EventProcessor::readRun() {
    if (principalCache_.hasRunPrincipal()) {
      throw edm::Exception(edm::errors::LogicError) << "EventProcessor::readRun\n"
                                                    << "Illegal attempt to insert run into cache\n"
                                                    << "Contact a Framework Developer\n";
    }
    auto rp = std::make_shared<RunPrincipal>(input_->runAuxiliary(),
                                             preg(),
                                             *processConfiguration_,
                                             historyAppender_.get(),
                                             0,
                                             true,
                                             &mergeableRunProductProcesses_);
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readRun(*rp, *historyAppender_);
      sentry.completedSuccessfully();
    }
    assert(input_->reducedProcessHistoryID() == rp->reducedProcessHistoryID());
    principalCache_.insert(rp);
    return std::make_pair(rp->reducedProcessHistoryID(), input_->run());
  }

  std::pair<ProcessHistoryID, RunNumber_t> EventProcessor::readAndMergeRun() {
    principalCache_.merge(input_->runAuxiliary(), preg());
    auto runPrincipal = principalCache_.runPrincipalPtr();
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
      throw edm::Exception(edm::errors::LogicError) << "EventProcessor::readLuminosityBlock\n"
                                                    << "Illegal attempt to insert lumi into cache\n"
                                                    << "Run is invalid\n"
                                                    << "Contact a Framework Developer\n";
    }
    auto lbp = principalCache_.getAvailableLumiPrincipalPtr();
    assert(lbp);
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
           input_->processHistoryRegistry().reducedProcessHistoryID(lumiPrincipal.aux().processHistoryID()) ==
               input_->processHistoryRegistry().reducedProcessHistoryID(
                   input_->luminosityBlockAuxiliary()->processHistoryID()));
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

  void EventProcessor::writeRunAsync(WaitingTaskHolder task,
                                     ProcessHistoryID const& phid,
                                     RunNumber_t run,
                                     MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    auto subsT = edm::make_waiting_task(
        tbb::task::allocate_root(),
        [this, phid, run, task, mergeableRunProductMetadata](std::exception_ptr const* iExcept) mutable {
          if (iExcept) {
            task.doneWaiting(*iExcept);
          } else {
            ServiceRegistry::Operate op(serviceToken_);
            for (auto& s : subProcesses_) {
              s.writeRunAsync(task, phid, run, mergeableRunProductMetadata);
            }
          }
        });
    ServiceRegistry::Operate op(serviceToken_);
    schedule_->writeRunAsync(WaitingTaskHolder(subsT),
                             principalCache_.runPrincipal(phid, run),
                             &processContext_,
                             actReg_.get(),
                             mergeableRunProductMetadata);
  }

  void EventProcessor::deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run) {
    principalCache_.deleteRun(phid, run);
    for_all(subProcesses_, [run, phid](auto& subProcess) { subProcess.deleteRunFromCache(phid, run); });
    FDEBUG(1) << "\tdeleteRunFromCache " << run << "\n";
  }

  void EventProcessor::writeLumiAsync(WaitingTaskHolder task, LuminosityBlockPrincipal& lumiPrincipal) {
    auto subsT = edm::make_waiting_task(tbb::task::allocate_root(),
                                        [this, task, &lumiPrincipal](std::exception_ptr const* iExcept) mutable {
                                          if (iExcept) {
                                            task.doneWaiting(*iExcept);
                                          } else {
                                            ServiceRegistry::Operate op(serviceToken_);
                                            for (auto& s : subProcesses_) {
                                              s.writeLumiAsync(task, lumiPrincipal);
                                            }
                                          }
                                        });
    ServiceRegistry::Operate op(serviceToken_);

    lumiPrincipal.runPrincipal().mergeableRunProductMetadata()->writeLumi(lumiPrincipal.luminosityBlock());

    schedule_->writeLumiAsync(WaitingTaskHolder{subsT}, lumiPrincipal, &processContext_, actReg_.get());
  }

  void EventProcessor::deleteLumiFromCache(LuminosityBlockProcessingStatus& iStatus) {
    for (auto& s : subProcesses_) {
      s.deleteLumiFromCache(*iStatus.lumiPrincipal());
    }
    iStatus.lumiPrincipal()->clearPrincipal();
    //FDEBUG(1) << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  bool EventProcessor::readNextEventForStream(unsigned int iStreamIndex, LuminosityBlockProcessingStatus& iStatus) {
    if (deferredExceptionPtrIsSet_.load(std::memory_order_acquire)) {
      iStatus.endLumi();
      return false;
    }

    if (iStatus.wasEventProcessingStopped()) {
      return false;
    }

    if (shouldWeStop()) {
      lastSourceTransition_ = InputSource::IsStop;
      iStatus.stopProcessingEvents();
      iStatus.endLumi();
      return false;
    }

    ServiceRegistry::Operate operate(serviceToken_);
    // Caught exception is propagated to EventProcessor::runToCompletion() via deferredExceptionPtr_
    CMS_SA_ALLOW try {
      //need to use lock in addition to the serial task queue because
      // of delayed provenance reading and reading data in response to
      // edm::Refs etc
      std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));

      auto itemType = iStatus.continuingLumi() ? InputSource::IsLumi : nextTransitionType();
      if (InputSource::IsLumi == itemType) {
        iStatus.haveContinuedLumi();
        while (itemType == InputSource::IsLumi and iStatus.lumiPrincipal()->run() == input_->run() and
               iStatus.lumiPrincipal()->luminosityBlock() == nextLuminosityBlockID()) {
          readAndMergeLumi(iStatus);
          itemType = nextTransitionType();
        }
        if (InputSource::IsLumi == itemType) {
          iStatus.setNextSyncValue(IOVSyncValue(EventID(input_->run(), input_->luminosityBlock(), 0),
                                                input_->luminosityBlockAuxiliary()->beginTime()));
        }
      }
      if (InputSource::IsEvent != itemType) {
        iStatus.stopProcessingEvents();

        //IsFile may continue processing the lumi and
        // looper_ can cause the input source to declare a new IsRun which is actually
        // just a continuation of the previous run
        if (InputSource::IsStop == itemType or InputSource::IsLumi == itemType or
            (InputSource::IsRun == itemType and iStatus.lumiPrincipal()->run() != input_->run())) {
          iStatus.endLumi();
        }
        return false;
      }
      readEvent(iStreamIndex);
    } catch (...) {
      bool expected = false;
      if (deferredExceptionPtrIsSet_.compare_exchange_strong(expected, true)) {
        deferredExceptionPtr_ = std::current_exception();
        iStatus.endLumi();
      }
      return false;
    }
    return true;
  }

  void EventProcessor::handleNextEventForStreamAsync(WaitingTaskHolder iTask, unsigned int iStreamIndex) {
    sourceResourcesAcquirer_.serialQueueChain().push([this, iTask, iStreamIndex]() mutable {
      ServiceRegistry::Operate operate(serviceToken_);
      auto& status = streamLumiStatus_[iStreamIndex];
      // Caught exception is propagated to EventProcessor::runToCompletion() via deferredExceptionPtr_
      CMS_SA_ALLOW try {
        if (readNextEventForStream(iStreamIndex, *status)) {
          auto recursionTask = make_waiting_task(
              tbb::task::allocate_root(), [this, iTask, iStreamIndex](std::exception_ptr const* iPtr) mutable {
                if (iPtr) {
                  // Try to end the stream properly even if an exception was
                  // thrown on an event.
                  bool expected = false;
                  if (deferredExceptionPtrIsSet_.compare_exchange_strong(expected, true)) {
                    // This is the case where the exception in iPtr is the primary
                    // exception and we want to see its message.
                    deferredExceptionPtr_ = *iPtr;
                    WaitingTaskHolder tempHolder(iTask);
                    tempHolder.doneWaiting(*iPtr);
                  }
                  streamEndLumiAsync(std::move(iTask), iStreamIndex, streamLumiStatus_[iStreamIndex]);
                  //the stream will stop now
                  return;
                }
                handleNextEventForStreamAsync(std::move(iTask), iStreamIndex);
              });

          processEventAsync(WaitingTaskHolder(recursionTask), iStreamIndex);
        } else {
          //the stream will stop now
          if (status->isLumiEnding()) {
            if (lastTransitionType() == InputSource::IsLumi and not status->haveStartedNextLumi()) {
              status->startNextLumi();
              beginLumiAsync(status->nextSyncValue(), status->runResource(), iTask);
            }
            streamEndLumiAsync(std::move(iTask), iStreamIndex, status);
          } else {
            iTask.doneWaiting(std::exception_ptr{});
          }
        }
      } catch (...) {
        // It is unlikely we will ever get in here ...
        // But if we do try to clean up and propagate the exception
        if (streamLumiStatus_[iStreamIndex]) {
          streamEndLumiAsync(iTask, iStreamIndex, streamLumiStatus_[iStreamIndex]);
        }
        bool expected = false;
        if (deferredExceptionPtrIsSet_.compare_exchange_strong(expected, true)) {
          auto e = std::current_exception();
          deferredExceptionPtr_ = e;
          iTask.doneWaiting(e);
        }
      }
    });
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

  void EventProcessor::processEventAsync(WaitingTaskHolder iHolder, unsigned int iStreamIndex) {
    tbb::task::spawn(
        *make_functor_task(tbb::task::allocate_root(), [=]() { processEventAsyncImpl(iHolder, iStreamIndex); }));
  }

  void EventProcessor::processEventAsyncImpl(WaitingTaskHolder iHolder, unsigned int iStreamIndex) {
    auto pep = &(principalCache_.eventPrincipal(iStreamIndex));

    ServiceRegistry::Operate operate(serviceToken_);
    Service<RandomNumberGenerator> rng;
    if (rng.isAvailable()) {
      Event ev(*pep, ModuleDescription(), nullptr);
      rng->postEventRead(ev);
    }

    WaitingTaskHolder finalizeEventTask(make_waiting_task(
        tbb::task::allocate_root(), [this, pep, iHolder, iStreamIndex](std::exception_ptr const* iPtr) mutable {
          //NOTE: If we have a looper we only have one Stream
          if (looper_) {
            ServiceRegistry::Operate operateLooper(serviceToken_);
            processEventWithLooper(*pep, iStreamIndex);
          }

          FDEBUG(1) << "\tprocessEvent\n";
          pep->clearEventPrincipal();
          if (iPtr) {
            iHolder.doneWaiting(*iPtr);
          } else {
            iHolder.doneWaiting(std::exception_ptr());
          }
        }));
    WaitingTaskHolder afterProcessTask;
    if (subProcesses_.empty()) {
      afterProcessTask = std::move(finalizeEventTask);
    } else {
      //Need to run SubProcesses after schedule has finished
      // with the event
      afterProcessTask = WaitingTaskHolder(make_waiting_task(
          tbb::task::allocate_root(),
          [this, pep, finalizeEventTask, iStreamIndex](std::exception_ptr const* iPtr) mutable {
            if (not iPtr) {
              //when run with 1 thread, we want to the order to be what
              // it was before. This requires reversing the order since
              // tasks are run last one in first one out
              for (auto& subProcess : boost::adaptors::reverse(subProcesses_)) {
                subProcess.doEventAsync(finalizeEventTask, *pep, &streamLumiStatus_[iStreamIndex]->eventSetupImpls());
              }
            } else {
              finalizeEventTask.doneWaiting(*iPtr);
            }
          }));
    }

    EventSetupImpl const& es = streamLumiStatus_[iStreamIndex]->eventSetupImpl(esp_->subProcessIndex());
    schedule_->processOneEventAsync(std::move(afterProcessTask), iStreamIndex, *pep, es, serviceToken_);
  }

  void EventProcessor::processEventWithLooper(EventPrincipal& iPrincipal, unsigned int iStreamIndex) {
    bool randomAccess = input_->randomAccess();
    ProcessingController::ForwardState forwardState = input_->forwardState();
    ProcessingController::ReverseState reverseState = input_->reverseState();
    ProcessingController pc(forwardState, reverseState, randomAccess);

    EDLooperBase::Status status = EDLooperBase::kContinue;
    do {
      StreamContext streamContext(iPrincipal.streamID(), &processContext_);
      EventSetupImpl const& es = streamLumiStatus_[iStreamIndex]->eventSetupImpl(esp_->subProcessIndex());
      status = looper_->doDuringLoop(iPrincipal, es, pc, &streamContext);

      bool succeeded = true;
      if (randomAccess) {
        if (pc.requestedTransition() == ProcessingController::kToPreviousEvent) {
          input_->skipEvents(-2);
        } else if (pc.requestedTransition() == ProcessingController::kToSpecifiedEvent) {
          succeeded = input_->goToEvent(pc.specifiedEventTransition());
        }
      }
      pc.setLastOperationSucceeded(succeeded);
    } while (!pc.lastOperationSucceeded());
    if (status != EDLooperBase::kContinue) {
      shouldWeStop_ = true;
      lastSourceTransition_ = InputSource::IsStop;
    }
  }

  bool EventProcessor::shouldWeStop() const {
    FDEBUG(1) << "\tshouldWeStop\n";
    if (shouldWeStop_)
      return true;
    if (!subProcesses_.empty()) {
      for (auto const& subProcess : subProcesses_) {
        if (subProcess.terminate()) {
          return true;
        }
      }
      return false;
    }
    return schedule_->terminate();
  }

  void EventProcessor::setExceptionMessageFiles(std::string& message) { exceptionMessageFiles_ = message; }

  void EventProcessor::setExceptionMessageRuns(std::string& message) { exceptionMessageRuns_ = message; }

  void EventProcessor::setExceptionMessageLumis() { exceptionMessageLumis_ = true; }

  bool EventProcessor::setDeferredException(std::exception_ptr iException) {
    bool expected = false;
    if (deferredExceptionPtrIsSet_.compare_exchange_strong(expected, true)) {
      deferredExceptionPtr_ = iException;
      return true;
    }
    return false;
  }

  void EventProcessor::warnAboutModulesRequiringLuminosityBLockSynchronization() const {
    std::unique_ptr<LogSystem> s;
    for (auto worker : schedule_->allWorkers()) {
      if (worker->wantsGlobalLuminosityBlocks() and worker->globalLuminosityBlocksQueue()) {
        if (not s) {
          s = std::make_unique<LogSystem>("ModulesSynchingOnLumis");
          (*s) << "The following modules require synchronizing on LuminosityBlock boundaries:";
        }
        (*s) << "\n  " << worker->description().moduleName() << " " << worker->description().moduleLabel();
      }
    }
  }
}  // namespace edm
