#include "FWCore/Framework/interface/EventProcessor.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"

#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Framework/src/CommonParams.h"
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
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/ScheduleItems.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/src/Breakpoints.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/Framework/interface/maker/InputSourceFactory.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/streamTransitionAsync.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/ensureAvailableAccelerators.h"
#include "FWCore/Framework/interface/globalTransitionAsync.h"

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
#include "FWCore/Concurrency/interface/chain_first.h"

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

#include "oneapi/tbb/task.h"

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

  namespace chain = waiting_task::chain;

  // ---------------------------------------------------------------
  std::unique_ptr<InputSource> makeInput(ParameterSet& params,
                                         CommonParams const& common,
                                         std::shared_ptr<ProductRegistry> preg,
                                         std::shared_ptr<BranchIDListHelper> branchIDListHelper,
                                         std::shared_ptr<ProcessBlockHelper> const& processBlockHelper,
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
                                  processBlockHelper,
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
  void validateLooper(ParameterSet& pset) {
    auto const modtype = pset.getParameter<std::string>("@module_type");
    auto const moduleLabel = pset.getParameter<std::string>("@module_label");
    auto filler = ParameterSetDescriptionFillerPluginFactory::get()->create(modtype);
    ConfigurationDescriptions descriptions(filler->baseType(), modtype);
    filler->fill(descriptions);
    try {
      edm::convertException::wrap([&]() { descriptions.validate(pset, moduleLabel); });
    } catch (cms::Exception& iException) {
      iException.addContext(
          fmt::format("Validating configuration of EDLooper of type {} with label: '{}'", modtype, moduleLabel));
      throw;
    }
  }

  std::shared_ptr<EDLooperBase> fillLooper(eventsetup::EventSetupsController& esController,
                                           eventsetup::EventSetupProvider& cp,
                                           ParameterSet& params,
                                           std::vector<std::string> const& loopers) {
    std::shared_ptr<EDLooperBase> vLooper;

    assert(1 == loopers.size());

    for (auto const& looperName : loopers) {
      ParameterSet* providerPSet = params.getPSetForUpdate(looperName);
      validateLooper(*providerPSet);
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
    ensureAvailableAccelerators(*parameterSet);

    //threading
    unsigned int nThreads = optionsPset.getUntrackedParameter<unsigned int>("numberOfThreads");

    // Even if numberOfThreads was set to zero in the Python configuration, the code
    // in cmsRun.cpp should have reset it to something else.
    assert(nThreads != 0);

    unsigned int nStreams = optionsPset.getUntrackedParameter<unsigned int>("numberOfStreams");
    if (nStreams == 0) {
      nStreams = nThreads;
    }
    unsigned int nConcurrentRuns = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentRuns");
    if (nConcurrentRuns != 1) {
      throw Exception(errors::Configuration, "Illegal value nConcurrentRuns : ")
          << "Although the plan is to change this in the future, currently nConcurrentRuns must always be 1.\n";
    }
    unsigned int nConcurrentLumis =
        optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentLuminosityBlocks");
    if (nConcurrentLumis == 0) {
      nConcurrentLumis = 2;
    }
    if (nConcurrentLumis > nStreams) {
      nConcurrentLumis = nStreams;
    }
    std::vector<std::string> loopers = parameterSet->getParameter<std::vector<std::string>>("@all_loopers");
    if (!loopers.empty()) {
      //For now loopers make us run only 1 transition at a time
      if (nStreams != 1 || nConcurrentLumis != 1 || nConcurrentRuns != 1) {
        edm::LogWarning("ThreadStreamSetup") << "There is a looper, so the number of streams, the number "
                                                "of concurrent runs, and the number of concurrent lumis "
                                                "are all being reset to 1. Loopers cannot currently support "
                                                "values greater than 1.";
        nStreams = 1;
        nConcurrentLumis = 1;
        nConcurrentRuns = 1;
      }
    }
    bool dumpOptions = optionsPset.getUntrackedParameter<bool>("dumpOptions");
    if (dumpOptions) {
      dumpOptionsToLogFile(nThreads, nStreams, nConcurrentLumis, nConcurrentRuns);
    } else {
      if (nThreads > 1 or nStreams > 1) {
        edm::LogInfo("ThreadStreamSetup") << "setting # threads " << nThreads << "\nsetting # streams " << nStreams;
      }
    }
    // The number of concurrent IOVs is configured individually for each record in
    // the class NumberOfConcurrentIOVs to values less than or equal to this.
    unsigned int maxConcurrentIOVs = nConcurrentLumis;

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
    deleteNonConsumedUnscheduledModules_ =
        optionsPset.getUntrackedParameter<bool>("deleteNonConsumedUnscheduledModules");
    //for now, if have a subProcess, don't allow early delete
    // In the future we should use the SubProcess's 'keep list' to decide what can be kept
    if (not hasSubProcesses) {
      branchesToDeleteEarly_ = optionsPset.getUntrackedParameter<std::vector<std::string>>("canDeleteEarly");
    }

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
    esp_ = espController_->makeProvider(
        *parameterSet, items.actReg_.get(), &eventSetupPset, maxConcurrentIOVs, dumpOptions);

    // initialize the looper, if any
    if (!loopers.empty()) {
      looper_ = fillLooper(*espController_, *esp_, *parameterSet, loopers);
      looper_->setActionTable(items.act_table_.get());
      looper_->attachTo(*items.actReg_);

      // in presence of looper do not delete modules
      deleteNonConsumedUnscheduledModules_ = false;
    }

    preallocations_ = PreallocationConfiguration{nThreads, nStreams, nConcurrentLumis, nConcurrentRuns};

    lumiQueue_ = std::make_unique<LimitedTaskQueue>(nConcurrentLumis);
    streamQueues_.resize(nStreams);
    streamLumiStatus_.resize(nStreams);

    processBlockHelper_ = std::make_shared<ProcessBlockHelper>();

    // initialize the input source
    input_ = makeInput(*parameterSet,
                       *common,
                       items.preg(),
                       items.branchIDListHelper(),
                       get_underlying_safe(processBlockHelper_),
                       items.thinnedAssociationsHelper(),
                       items.actReg_,
                       items.processConfiguration(),
                       preallocations_);

    // initialize the Schedule
    schedule_ =
        items.initSchedule(*parameterSet, hasSubProcesses, preallocations_, &processContext_, *processBlockHelper_);

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
                                                 index,
                                                 true /*primary process*/,
                                                 &*processBlockHelper_);
      principalCache_.insert(std::move(ep));
    }

    for (unsigned int index = 0; index < preallocations_.numberOfLuminosityBlocks(); ++index) {
      auto lp =
          std::make_unique<LuminosityBlockPrincipal>(preg(), *processConfiguration_, historyAppender_.get(), index);
      principalCache_.insert(std::move(lp));
    }

    {
      auto pb = std::make_unique<ProcessBlockPrincipal>(preg(), *processConfiguration_);
      principalCache_.insert(std::move(pb));

      auto pbForInput = std::make_unique<ProcessBlockPrincipal>(preg(), *processConfiguration_);
      principalCache_.insertForInput(std::move(pbForInput));
    }

    // fill the subprocesses, if there are any
    subProcesses_.reserve(subProcessVParameterSet.size());
    for (auto& subProcessPSet : subProcessVParameterSet) {
      subProcesses_.emplace_back(subProcessPSet,
                                 *parameterSet,
                                 preg(),
                                 branchIDListHelper(),
                                 *processBlockHelper_,
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

  void EventProcessor::taskCleanup() {
    edm::FinalWaitingTask task;
    espController_->endIOVsAsync(edm::WaitingTaskHolder{taskGroup_, &task});
    taskGroup_.wait();
    assert(task.done());
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

    std::vector<ModuleProcessName> consumedBySubProcesses;
    for_all(subProcesses_,
            [&consumedBySubProcesses, deleteModules = deleteNonConsumedUnscheduledModules_](auto& subProcess) {
              auto c = subProcess.keepOnlyConsumedUnscheduledModules(deleteModules);
              if (consumedBySubProcesses.empty()) {
                consumedBySubProcesses = std::move(c);
              } else if (not c.empty()) {
                std::vector<ModuleProcessName> tmp;
                tmp.reserve(consumedBySubProcesses.size() + c.size());
                std::merge(consumedBySubProcesses.begin(),
                           consumedBySubProcesses.end(),
                           c.begin(),
                           c.end(),
                           std::back_inserter(tmp));
                std::swap(consumedBySubProcesses, tmp);
              }
            });

    // Note: all these may throw
    checkForModuleDependencyCorrectness(pathsAndConsumesOfModules_, printDependencies_);
    if (deleteNonConsumedUnscheduledModules_) {
      if (auto const unusedModules = nonConsumedUnscheduledModules(pathsAndConsumesOfModules_, consumedBySubProcesses);
          not unusedModules.empty()) {
        pathsAndConsumesOfModules_.removeModules(unusedModules);

        edm::LogInfo("DeleteModules").log([&unusedModules](auto& l) {
          l << "Following modules are not in any Path or EndPath, nor is their output consumed by any other module, "
               "and "
               "therefore they are deleted before beginJob transition.";
          for (auto const& description : unusedModules) {
            l << "\n " << description->moduleLabel();
          }
        });
        for (auto const& description : unusedModules) {
          schedule_->deleteModule(description->moduleLabel(), actReg_.get());
        }
      }
    }
    // Initialize after the deletion of non-consumed unscheduled
    // modules to avoid non-consumed non-run modules to keep the
    // products unnecessarily alive
    if (not branchesToDeleteEarly_.empty()) {
      schedule_->initializeEarlyDelete(branchesToDeleteEarly_, *preg_);
      decltype(branchesToDeleteEarly_)().swap(branchesToDeleteEarly_);
    }

    actReg_->preBeginJobSignal_(pathsAndConsumesOfModules_, processContext_);

    if (preallocations_.numberOfLuminosityBlocks() > 1) {
      throwAboutModulesRequiringLuminosityBlockSynchronization();
    }
    warnAboutLegacyModules();

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
    schedule_->beginJob(*preg_, esp_->recordsToProxyIndices(), *processBlockHelper_);
    if (looper_) {
      constexpr bool mustPrefetchMayGet = true;
      auto const processBlockLookup = preg_->productLookup(InProcess);
      auto const runLookup = preg_->productLookup(InRun);
      auto const lumiLookup = preg_->productLookup(InLumi);
      auto const eventLookup = preg_->productLookup(InEvent);
      looper_->updateLookup(InProcess, *processBlockLookup, mustPrefetchMayGet);
      looper_->updateLookup(InRun, *runLookup, mustPrefetchMayGet);
      looper_->updateLookup(InLumi, *lumiLookup, mustPrefetchMayGet);
      looper_->updateLookup(InEvent, *eventLookup, mustPrefetchMayGet);
      looper_->updateLookup(esp_->recordsToProxyIndices());
    }
    // toerror.succeeded(); // should we add this?
    for_all(subProcesses_, [](auto& subProcess) { subProcess.doBeginJob(); });
    actReg_->postBeginJobSignal_();

    FinalWaitingTask last;
    oneapi::tbb::task_group group;
    using namespace edm::waiting_task::chain;
    first([this](auto nextTask) {
      for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
        first([i, this](auto nextTask) {
          ServiceRegistry::Operate operate(serviceToken_);
          schedule_->beginStream(i);
        }) | ifThen(not subProcesses_.empty(), [this, i](auto nextTask) {
          ServiceRegistry::Operate operate(serviceToken_);
          for_all(subProcesses_, [i](auto& subProcess) { subProcess.doBeginStream(i); });
        }) | lastTask(nextTask);
      }
    }) | runLast(WaitingTaskHolder(group, &last));
    group.wait();
    if (last.exceptionPtr()) {
      std::rethrow_exception(*last.exceptionPtr());
    }
  }

  void EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c(
        "Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    using namespace edm::waiting_task::chain;

    edm::FinalWaitingTask waitTask;
    oneapi::tbb::task_group group;

    {
      //handle endStream transitions
      edm::WaitingTaskHolder taskHolder(group, &waitTask);
      std::mutex collectorMutex;
      for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
        first([this, i, &c, &collectorMutex](auto nextTask) {
          std::exception_ptr ep;
          try {
            ServiceRegistry::Operate operate(serviceToken_);
            this->schedule_->endStream(i);
          } catch (...) {
            ep = std::current_exception();
          }
          if (ep) {
            std::lock_guard<std::mutex> l(collectorMutex);
            c.call([&ep]() { std::rethrow_exception(ep); });
          }
        }) | then([this, i, &c, &collectorMutex](auto nextTask) {
          for (auto& subProcess : subProcesses_) {
            first([this, i, &c, &collectorMutex, &subProcess](auto nextTask) {
              std::exception_ptr ep;
              try {
                ServiceRegistry::Operate operate(serviceToken_);
                subProcess.doEndStream(i);
              } catch (...) {
                ep = std::current_exception();
              }
              if (ep) {
                std::lock_guard<std::mutex> l(collectorMutex);
                c.call([&ep]() { std::rethrow_exception(ep); });
              }
            }) | lastTask(nextTask);
          }
        }) | lastTask(taskHolder);
      }
    }
    group.wait();

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
    if (fileBlockValid()) {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->closeFile(fb_.get(), cleaningUpAfterException);
      sentry.completedSuccessfully();
    }
    FDEBUG(1) << "\tcloseInputFile\n";
  }

  void EventProcessor::openOutputFiles() {
    if (fileBlockValid()) {
      schedule_->openOutputFiles(*fb_);
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.openOutputFiles(*fb_); });
    }
    FDEBUG(1) << "\topenOutputFiles\n";
  }

  void EventProcessor::closeOutputFiles() {
    schedule_->closeOutputFiles();
    for_all(subProcesses_, [](auto& subProcess) { subProcess.closeOutputFiles(); });
    processBlockHelper_->clearAfterOutputFilesClose();
    FDEBUG(1) << "\tcloseOutputFiles\n";
  }

  void EventProcessor::respondToOpenInputFile() {
    if (fileBlockValid()) {
      for_all(subProcesses_,
              [this](auto& subProcess) { subProcess.updateBranchIDListHelper(branchIDListHelper_->branchIDLists()); });
      schedule_->respondToOpenInputFile(*fb_);
      for_all(subProcesses_, [this](auto& subProcess) { subProcess.respondToOpenInputFile(*fb_); });
    }
    FDEBUG(1) << "\trespondToOpenInputFile\n";
  }

  void EventProcessor::respondToCloseInputFile() {
    if (fileBlockValid()) {
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

  void EventProcessor::beginProcessBlock(bool& beginProcessBlockSucceeded) {
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();
    processBlockPrincipal.fillProcessBlockPrincipal(processConfiguration_->processName());

    using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>;
    FinalWaitingTask globalWaitTask;

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    beginGlobalTransitionAsync<Traits>(
        WaitingTaskHolder(taskGroup_, &globalWaitTask), *schedule_, transitionInfo, serviceToken_, subProcesses_);

    do {
      taskGroup_.wait();
    } while (not globalWaitTask.done());

    if (globalWaitTask.exceptionPtr() != nullptr) {
      std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
    }
    beginProcessBlockSucceeded = true;
  }

  void EventProcessor::inputProcessBlocks() {
    input_->fillProcessBlockHelper();
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
    while (input_->nextProcessBlock(processBlockPrincipal)) {
      readProcessBlock(processBlockPrincipal);

      using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>;
      FinalWaitingTask globalWaitTask;

      ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
      beginGlobalTransitionAsync<Traits>(
          WaitingTaskHolder(taskGroup_, &globalWaitTask), *schedule_, transitionInfo, serviceToken_, subProcesses_);

      do {
        taskGroup_.wait();
      } while (not globalWaitTask.done());
      if (globalWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
      }

      FinalWaitingTask writeWaitTask;
      writeProcessBlockAsync(edm::WaitingTaskHolder{taskGroup_, &writeWaitTask}, ProcessBlockType::Input);
      do {
        taskGroup_.wait();
      } while (not writeWaitTask.done());
      if (writeWaitTask.exceptionPtr()) {
        std::rethrow_exception(*writeWaitTask.exceptionPtr());
      }

      processBlockPrincipal.clearPrincipal();
      for (auto& s : subProcesses_) {
        s.clearProcessBlockPrincipal(ProcessBlockType::Input);
      }
    }
  }

  void EventProcessor::endProcessBlock(bool cleaningUpAfterException, bool beginProcessBlockSucceeded) {
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();

    using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
    FinalWaitingTask globalWaitTask;

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    endGlobalTransitionAsync<Traits>(WaitingTaskHolder(taskGroup_, &globalWaitTask),
                                     *schedule_,
                                     transitionInfo,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
    do {
      taskGroup_.wait();
    } while (not globalWaitTask.done());
    if (globalWaitTask.exceptionPtr() != nullptr) {
      std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
    }

    if (beginProcessBlockSucceeded) {
      FinalWaitingTask writeWaitTask;
      writeProcessBlockAsync(edm::WaitingTaskHolder{taskGroup_, &writeWaitTask}, ProcessBlockType::New);
      do {
        taskGroup_.wait();
      } while (not writeWaitTask.done());
      if (writeWaitTask.exceptionPtr()) {
        std::rethrow_exception(*writeWaitTask.exceptionPtr());
      }
    }

    processBlockPrincipal.clearPrincipal();
    for (auto& s : subProcesses_) {
      s.clearProcessBlockPrincipal(ProcessBlockType::New);
    }
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
      actReg_->preESSyncIOVSignal_.emit(ts);
      synchronousEventSetupForInstance(ts, taskGroup_, *espController_);
      actReg_->postESSyncIOVSignal_.emit(ts);
      eventSetupForInstanceSucceeded = true;
      sentry.completedSuccessfully();
    }
    auto const& es = esp_->eventSetupImpl();
    if (looper_ && looperBeginJobRun_ == false) {
      looper_->copyInfo(ScheduleInfo(schedule_.get()));

      FinalWaitingTask waitTask;
      using namespace edm::waiting_task::chain;
      chain::first([this, &es](auto nextTask) {
        looper_->esPrefetchAsync(nextTask, es, Transition::BeginRun, serviceToken_);
      }) | then([this, &es](auto nextTask) mutable {
        looper_->beginOfJob(es);
        looperBeginJobRun_ = true;
        looper_->doStartingNewLoop();
      }) | runLast(WaitingTaskHolder(taskGroup_, &waitTask));

      do {
        taskGroup_.wait();
      } while (not waitTask.done());
      if (waitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(waitTask.exceptionPtr()));
      }
    }
    {
      using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>;
      FinalWaitingTask globalWaitTask;

      using namespace edm::waiting_task::chain;
      chain::first([&runPrincipal, &es, this](auto waitTask) {
        RunTransitionInfo transitionInfo(runPrincipal, es);
        beginGlobalTransitionAsync<Traits>(
            std::move(waitTask), *schedule_, transitionInfo, serviceToken_, subProcesses_);
      }) | then([&globalBeginSucceeded, run](auto waitTask) mutable {
        globalBeginSucceeded = true;
        FDEBUG(1) << "\tbeginRun " << run << "\n";
      }) | ifThen(looper_, [this, &runPrincipal, &es](auto waitTask) {
        looper_->prefetchAsync(waitTask, serviceToken_, Transition::BeginRun, runPrincipal, es);
      }) | ifThen(looper_, [this, &runPrincipal, &es](auto waitTask) {
        looper_->doBeginRun(runPrincipal, es, &processContext_);
      }) | runLast(WaitingTaskHolder(taskGroup_, &globalWaitTask));

      do {
        taskGroup_.wait();
      } while (not globalWaitTask.done());
      if (globalWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
      }
    }
    {
      //To wait, the ref count has to be 1+#streams
      FinalWaitingTask streamLoopWaitTask;

      using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>;

      RunTransitionInfo transitionInfo(runPrincipal, es);
      beginStreamsTransitionAsync<Traits>(WaitingTaskHolder(taskGroup_, &streamLoopWaitTask),
                                          *schedule_,
                                          preallocations_.numberOfStreams(),
                                          transitionInfo,
                                          serviceToken_,
                                          subProcesses_);
      do {
        taskGroup_.wait();
      } while (not streamLoopWaitTask.done());
      if (streamLoopWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(streamLoopWaitTask.exceptionPtr()));
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
        RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, run);
        if (runPrincipal.shouldWriteRun() != RunPrincipal::kNo) {
          FinalWaitingTask t;
          MergeableRunProductMetadata* mergeableRunProductMetadata = runPrincipal.mergeableRunProductMetadata();
          mergeableRunProductMetadata->preWriteRun();
          writeRunAsync(edm::WaitingTaskHolder{taskGroup_, &t}, phid, run, mergeableRunProductMetadata);
          do {
            taskGroup_.wait();
          } while (not t.done());
          mergeableRunProductMetadata->postWriteRun();
          if (t.exceptionPtr()) {
            std::rethrow_exception(*t.exceptionPtr());
          }
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
      actReg_->preESSyncIOVSignal_.emit(ts);
      synchronousEventSetupForInstance(ts, taskGroup_, *espController_);
      actReg_->postESSyncIOVSignal_.emit(ts);
      sentry.completedSuccessfully();
    }
    auto const& es = esp_->eventSetupImpl();
    if (globalBeginSucceeded) {
      //To wait, the ref count has to be 1+#streams
      FinalWaitingTask streamLoopWaitTask;

      using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>;

      RunTransitionInfo transitionInfo(runPrincipal, es);
      endStreamsTransitionAsync<Traits>(WaitingTaskHolder(taskGroup_, &streamLoopWaitTask),
                                        *schedule_,
                                        preallocations_.numberOfStreams(),
                                        transitionInfo,
                                        serviceToken_,
                                        subProcesses_,
                                        cleaningUpAfterException);
      do {
        taskGroup_.wait();
      } while (not streamLoopWaitTask.done());
      if (streamLoopWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(streamLoopWaitTask.exceptionPtr()));
      }
    }
    FDEBUG(1) << "\tstreamEndRun " << run << "\n";
    if (looper_) {
      //looper_->doStreamEndRun(schedule_->streamID(),runPrincipal, es);
    }
    {
      FinalWaitingTask globalWaitTask;

      using namespace edm::waiting_task::chain;
      chain::first([this, &runPrincipal, &es, cleaningUpAfterException](auto nextTask) {
        RunTransitionInfo transitionInfo(runPrincipal, es);
        using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>;
        endGlobalTransitionAsync<Traits>(
            std::move(nextTask), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
      }) | ifThen(looper_, [this, &runPrincipal, &es](auto nextTask) {
        looper_->prefetchAsync(std::move(nextTask), serviceToken_, Transition::EndRun, runPrincipal, es);
      }) | ifThen(looper_, [this, &runPrincipal, &es](auto nextTask) {
        looper_->doEndRun(runPrincipal, es, &processContext_);
      }) | runLast(WaitingTaskHolder(taskGroup_, &globalWaitTask));

      do {
        taskGroup_.wait();
      } while (not globalWaitTask.done());
      if (globalWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
      }
    }
    FDEBUG(1) << "\tendRun " << run << "\n";
  }

  InputSource::ItemType EventProcessor::processLumis(std::shared_ptr<void> const& iRunResource) {
    FinalWaitingTask waitTask;
    if (streamLumiActive_ > 0) {
      assert(streamLumiActive_ == preallocations_.numberOfStreams());
      // Continue after opening a new input file
      continueLumiAsync(WaitingTaskHolder{taskGroup_, &waitTask});
    } else {
      beginLumiAsync(IOVSyncValue(EventID(input_->run(), input_->luminosityBlock(), 0),
                                  input_->luminosityBlockAuxiliary()->beginTime()),
                     iRunResource,
                     WaitingTaskHolder{taskGroup_, &waitTask});
    }
    do {
      taskGroup_.wait();
    } while (not waitTask.done());

    if (waitTask.exceptionPtr() != nullptr) {
      std::rethrow_exception(*(waitTask.exceptionPtr()));
    }
    return lastTransitionType();
  }

  void EventProcessor::beginLumiAsync(IOVSyncValue const& iSync,
                                      std::shared_ptr<void> const& iRunResource,
                                      edm::WaitingTaskHolder iHolder) {
    if (iHolder.taskHasFailed()) {
      return;
    }

    actReg_->esSyncIOVQueuingSignal_.emit(iSync);
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
    chain::first([&](auto nextTask) {
      auto asyncEventSetup = [](ActivityRegistry* actReg,
                                auto* espController,
                                auto& queue,
                                WaitingTaskHolder task,
                                auto& status,
                                IOVSyncValue const& iSync) {
        queue.pause();
        CMS_SA_ALLOW try {
          SendSourceTerminationSignalIfException sentry(actReg);
          // Pass in iSync to let the EventSetup system know which run and lumi
          // need to be processed and prepare IOVs for it.
          // Pass in the endIOVWaitingTasks so the lumi can notify them when the
          // lumi is done and no longer needs its EventSetup IOVs.
          actReg->preESSyncIOVSignal_.emit(iSync);
          espController->eventSetupForInstanceAsync(
              iSync, task, status->endIOVWaitingTasks(), status->eventSetupImpls());
          sentry.completedSuccessfully();
        } catch (...) {
          task.doneWaiting(std::current_exception());
        }
      };
      if (espController_->doWeNeedToWaitForIOVsToFinish(iSync)) {
        // We only get here inside this block if there is an EventSetup
        // module not able to handle concurrent IOVs (usually an ESSource)
        // and the new sync value is outside the current IOV of that module.
        auto group = nextTask.group();
        queueWhichWaitsForIOVsToFinish_.push(
            *group, [this, task = std::move(nextTask), iSync, status, asyncEventSetup]() mutable {
              asyncEventSetup(
                  actReg_.get(), espController_.get(), queueWhichWaitsForIOVsToFinish_, std::move(task), status, iSync);
            });
      } else {
        asyncEventSetup(
            actReg_.get(), espController_.get(), queueWhichWaitsForIOVsToFinish_, std::move(nextTask), status, iSync);
      }
    }) | chain::then([this, status, iSync](std::exception_ptr const* iPtr, auto nextTask) {
      actReg_->postESSyncIOVSignal_.emit(iSync);
      //the call to doneWaiting will cause the count to decrement
      auto copyTask = nextTask;
      if (iPtr) {
        nextTask.doneWaiting(*iPtr);
      }
      auto group = copyTask.group();
      lumiQueue_->pushAndPause(
          *group, [this, task = std::move(copyTask), status](edm::LimitedTaskQueue::Resumer iResumer) mutable {
            if (task.taskHasFailed()) {
              status->resetResources();
              return;
            }

            status->setResumer(std::move(iResumer));

            auto group = task.group();
            sourceResourcesAcquirer_.serialQueueChain().push(
                *group, [this, postQueueTask = std::move(task), status = std::move(status)]() mutable {
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

                    EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());

                    using namespace edm::waiting_task::chain;
                    chain::first([this, status, &lumiPrincipal](auto nextTask) {
                      EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());
                      {
                        LumiTransitionInfo transitionInfo(lumiPrincipal, es, &status->eventSetupImpls());
                        using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>;
                        beginGlobalTransitionAsync<Traits>(
                            nextTask, *schedule_, transitionInfo, serviceToken_, subProcesses_);
                      }
                    }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                      looper_->prefetchAsync(
                          nextTask, serviceToken_, Transition::BeginLuminosityBlock, *(status->lumiPrincipal()), es);
                    }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                      status->globalBeginDidSucceed();
                      //make the services available
                      ServiceRegistry::Operate operateLooper(serviceToken_);
                      looper_->doBeginLuminosityBlock(*(status->lumiPrincipal()), es, &processContext_);
                    }) | then([this, status](std::exception_ptr const* iPtr, auto holder) mutable {
                      if (iPtr) {
                        status->resetResources();
                        holder.doneWaiting(*iPtr);
                      } else {
                        if (not looper_) {
                          status->globalBeginDidSucceed();
                        }
                        EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());
                        using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>;

                        for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
                          streamQueues_[i].push(*holder.group(), [this, i, status, holder, &es]() mutable {
                            streamQueues_[i].pause();

                            auto& event = principalCache_.eventPrincipal(i);
                            //We need to be sure that 'status' and its internal shared_ptr<LuminosityBlockPrincipal> are only
                            // held by the container as this lambda may not finish executing before all the tasks it
                            // spawns have already started to run.
                            auto eventSetupImpls = &status->eventSetupImpls();
                            auto lp = status->lumiPrincipal().get();
                            streamLumiStatus_[i] = std::move(status);
                            ++streamLumiActive_;
                            event.setLuminosityBlockPrincipal(lp);
                            LumiTransitionInfo transitionInfo(*lp, es, eventSetupImpls);
                            using namespace edm::waiting_task::chain;
                            chain::first([this, i, &transitionInfo](auto nextTask) {
                              beginStreamTransitionAsync<Traits>(
                                  std::move(nextTask), *schedule_, i, transitionInfo, serviceToken_, subProcesses_);
                            }) | then([this, i](std::exception_ptr const* exceptionFromBeginStreamLumi, auto nextTask) {
                              if (exceptionFromBeginStreamLumi) {
                                WaitingTaskHolder tmp(nextTask);
                                tmp.doneWaiting(*exceptionFromBeginStreamLumi);
                                streamEndLumiAsync(nextTask, i);
                              } else {
                                handleNextEventForStreamAsync(std::move(nextTask), i);
                              }
                            }) | runLast(holder);
                          });
                        }
                      }
                    }) | runLast(postQueueTask);

                  } catch (...) {
                    status->resetResources();
                    postQueueTask.doneWaiting(std::current_exception());
                  }
                });  // task in sourceResourcesAcquirer
          });
    }) | chain::runLast(std::move(iHolder));
  }

  void EventProcessor::continueLumiAsync(edm::WaitingTaskHolder iHolder) {
    {
      //all streams are sharing the same status at the moment
      auto status = streamLumiStatus_[0];  //read from streamLumiActive_ happened in calling routine
      status->needToContinueLumi();
      status->startProcessingEvents();
    }

    unsigned int streamIndex = 0;
    oneapi::tbb::task_arena arena{oneapi::tbb::task_arena::attach()};
    for (; streamIndex < preallocations_.numberOfStreams() - 1; ++streamIndex) {
      arena.enqueue([this, streamIndex, h = iHolder]() { handleNextEventForStreamAsync(h, streamIndex); });
    }
    iHolder.group()->run(
        [this, streamIndex, h = std::move(iHolder)]() { handleNextEventForStreamAsync(h, streamIndex); });
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

    using namespace edm::waiting_task::chain;
    chain::first([this, &lp, &es, &eventSetupImpls, cleaningUpAfterException](auto nextTask) {
      IOVSyncValue ts(EventID(lp.run(), lp.luminosityBlock(), EventID::maxEventNumber()), lp.beginTime());

      LumiTransitionInfo transitionInfo(lp, es, eventSetupImpls);
      using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>;
      endGlobalTransitionAsync<Traits>(
          std::move(nextTask), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
    }) | then([this, didGlobalBeginSucceed, &lumiPrincipal = lp](auto nextTask) {
      //Only call writeLumi if beginLumi succeeded
      if (didGlobalBeginSucceed) {
        writeLumiAsync(std::move(nextTask), lumiPrincipal);
      }
    }) | ifThen(looper_, [this, &lp, &es](auto nextTask) {
      looper_->prefetchAsync(std::move(nextTask), serviceToken_, Transition::EndLuminosityBlock, lp, es);
    }) | ifThen(looper_, [this, &lp, &es](auto nextTask) {
      //any thrown exception auto propagates to nextTask via the chain
      ServiceRegistry::Operate operate(serviceToken_);
      looper_->doEndLuminosityBlock(lp, es, &processContext_);
    }) | then([status = std::move(iLumiStatus), this](std::exception_ptr const* iPtr, auto nextTask) mutable {
      std::exception_ptr ptr;
      if (iPtr) {
        ptr = *iPtr;
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
        handleEndLumiExceptions(&ptr, nextTask);
      }
    }) | runLast(std::move(iTask));
  }

  void EventProcessor::streamEndLumiAsync(edm::WaitingTaskHolder iTask, unsigned int iStreamIndex) {
    auto t = edm::make_waiting_task([this, iStreamIndex, iTask](std::exception_ptr const* iPtr) mutable {
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

    edm::WaitingTaskHolder lumiDoneTask{*iTask.group(), t};

    //Need to be sure the lumi status is released before lumiDoneTask can every be called.
    // therefore we do not want to hold the shared_ptr
    auto lumiStatus = streamLumiStatus_[iStreamIndex].get();
    lumiStatus->setEndTime();

    EventSetupImpl const& es = lumiStatus->eventSetupImpl(esp_->subProcessIndex());

    bool cleaningUpAfterException = lumiStatus->cleaningUpAfterException();
    auto eventSetupImpls = &lumiStatus->eventSetupImpls();

    if (lumiStatus->didGlobalBeginSucceed()) {
      auto& lumiPrincipal = *lumiStatus->lumiPrincipal();
      IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), EventID::maxEventNumber()),
                      lumiPrincipal.endTime());
      using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;
      LumiTransitionInfo transitionInfo(lumiPrincipal, es, eventSetupImpls);
      endStreamTransitionAsync<Traits>(std::move(lumiDoneTask),
                                       *schedule_,
                                       iStreamIndex,
                                       transitionInfo,
                                       serviceToken_,
                                       subProcesses_,
                                       cleaningUpAfterException);
    }
  }

  void EventProcessor::endUnfinishedLumi() {
    if (streamLumiActive_.load() > 0) {
      FinalWaitingTask globalWaitTask;
      {
        WaitingTaskHolder globalTaskHolder{taskGroup_, &globalWaitTask};
        for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
          if (streamLumiStatus_[i]) {
            streamEndLumiAsync(globalTaskHolder, i);
          }
        }
      }
      do {
        taskGroup_.wait();
      } while (not globalWaitTask.done());
      if (globalWaitTask.exceptionPtr() != nullptr) {
        std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
      }
    }
  }

  void EventProcessor::readProcessBlock(ProcessBlockPrincipal& processBlockPrincipal) {
    SendSourceTerminationSignalIfException sentry(actReg_.get());
    input_->readProcessBlock(processBlockPrincipal);
    sentry.completedSuccessfully();
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

  void EventProcessor::writeProcessBlockAsync(WaitingTaskHolder task, ProcessBlockType processBlockType) {
    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      ServiceRegistry::Operate op(serviceToken_);
      schedule_->writeProcessBlockAsync(
          nextTask, principalCache_.processBlockPrincipal(processBlockType), &processContext_, actReg_.get());
    }) | chain::ifThen(not subProcesses_.empty(), [this, processBlockType](auto nextTask) {
      ServiceRegistry::Operate op(serviceToken_);
      for (auto& s : subProcesses_) {
        s.writeProcessBlockAsync(nextTask, processBlockType);
      }
    }) | chain::runLast(std::move(task));
  }

  void EventProcessor::writeRunAsync(WaitingTaskHolder task,
                                     ProcessHistoryID const& phid,
                                     RunNumber_t run,
                                     MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      ServiceRegistry::Operate op(serviceToken_);
      schedule_->writeRunAsync(nextTask,
                               principalCache_.runPrincipal(phid, run),
                               &processContext_,
                               actReg_.get(),
                               mergeableRunProductMetadata);
    }) | chain::ifThen(not subProcesses_.empty(), [this, phid, run, mergeableRunProductMetadata](auto nextTask) {
      ServiceRegistry::Operate op(serviceToken_);
      for (auto& s : subProcesses_) {
        s.writeRunAsync(nextTask, phid, run, mergeableRunProductMetadata);
      }
    }) | chain::runLast(std::move(task));
  }

  void EventProcessor::deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run) {
    principalCache_.deleteRun(phid, run);
    for_all(subProcesses_, [run, phid](auto& subProcess) { subProcess.deleteRunFromCache(phid, run); });
    FDEBUG(1) << "\tdeleteRunFromCache " << run << "\n";
  }

  void EventProcessor::writeLumiAsync(WaitingTaskHolder task, LuminosityBlockPrincipal& lumiPrincipal) {
    using namespace edm::waiting_task;
    if (lumiPrincipal.shouldWriteLumi() != LuminosityBlockPrincipal::kNo) {
      chain::first([&](auto nextTask) {
        ServiceRegistry::Operate op(serviceToken_);

        lumiPrincipal.runPrincipal().mergeableRunProductMetadata()->writeLumi(lumiPrincipal.luminosityBlock());
        schedule_->writeLumiAsync(nextTask, lumiPrincipal, &processContext_, actReg_.get());
      }) | chain::ifThen(not subProcesses_.empty(), [this, &lumiPrincipal](auto nextTask) {
        ServiceRegistry::Operate op(serviceToken_);
        for (auto& s : subProcesses_) {
          s.writeLumiAsync(nextTask, lumiPrincipal);
        }
      }) | chain::lastTask(std::move(task));
    }
  }

  void EventProcessor::deleteLumiFromCache(LuminosityBlockProcessingStatus& iStatus) {
    for (auto& s : subProcesses_) {
      s.deleteLumiFromCache(*iStatus.lumiPrincipal());
    }
    iStatus.lumiPrincipal()->setShouldWriteLumi(LuminosityBlockPrincipal::kUninitialized);
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
    sourceResourcesAcquirer_.serialQueueChain().push(*iTask.group(), [this, iTask, iStreamIndex]() mutable {
      ServiceRegistry::Operate operate(serviceToken_);
      //we do not want to extend the lifetime of the shared_ptr to the end of this function
      // as steramEndLumiAsync may clear the value from streamLumiStatus_[iStreamIndex]
      auto status = streamLumiStatus_[iStreamIndex].get();
      // Caught exception is propagated to EventProcessor::runToCompletion() via deferredExceptionPtr_
      CMS_SA_ALLOW try {
        if (readNextEventForStream(iStreamIndex, *status)) {
          auto recursionTask = make_waiting_task([this, iTask, iStreamIndex](std::exception_ptr const* iPtr) mutable {
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
              streamEndLumiAsync(std::move(iTask), iStreamIndex);
              //the stream will stop now
              return;
            }
            handleNextEventForStreamAsync(std::move(iTask), iStreamIndex);
          });

          processEventAsync(WaitingTaskHolder(*iTask.group(), recursionTask), iStreamIndex);
        } else {
          //the stream will stop now
          if (status->isLumiEnding()) {
            if (lastTransitionType() == InputSource::IsLumi and not status->haveStartedNextLumi()) {
              status->startNextLumi();
              beginLumiAsync(status->nextSyncValue(), status->runResource(), iTask);
            }
            streamEndLumiAsync(std::move(iTask), iStreamIndex);
          } else {
            iTask.doneWaiting(std::exception_ptr{});
          }
        }
      } catch (...) {
        // It is unlikely we will ever get in here ...
        // But if we do try to clean up and propagate the exception
        if (streamLumiStatus_[iStreamIndex]) {
          streamEndLumiAsync(iTask, iStreamIndex);
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
    iHolder.group()->run([=]() { processEventAsyncImpl(iHolder, iStreamIndex); });
  }

  void EventProcessor::processEventAsyncImpl(WaitingTaskHolder iHolder, unsigned int iStreamIndex) {
    auto pep = &(principalCache_.eventPrincipal(iStreamIndex));

    ServiceRegistry::Operate operate(serviceToken_);
    Service<RandomNumberGenerator> rng;
    if (rng.isAvailable()) {
      Event ev(*pep, ModuleDescription(), nullptr);
      rng->postEventRead(ev);
    }

    EventSetupImpl const& es = streamLumiStatus_[iStreamIndex]->eventSetupImpl(esp_->subProcessIndex());
    using namespace edm::waiting_task::chain;
    chain::first([this, &es, pep, iStreamIndex](auto nextTask) {
      EventTransitionInfo info(*pep, es);
      schedule_->processOneEventAsync(std::move(nextTask), iStreamIndex, info, serviceToken_);
    }) | ifThen(not subProcesses_.empty(), [this, pep, iStreamIndex](auto nextTask) {
      for (auto& subProcess : boost::adaptors::reverse(subProcesses_)) {
        subProcess.doEventAsync(nextTask, *pep, &streamLumiStatus_[iStreamIndex]->eventSetupImpls());
      }
    }) | ifThen(looper_, [this, iStreamIndex, pep](auto nextTask) {
      //NOTE: behavior change. previously if an exception happened looper was still called. Now it will not be called
      ServiceRegistry::Operate operateLooper(serviceToken_);
      processEventWithLooper(*pep, iStreamIndex);
    }) | then([pep](auto nextTask) {
      FDEBUG(1) << "\tprocessEvent\n";
      pep->clearEventPrincipal();
    }) | runLast(iHolder);
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

  void EventProcessor::throwAboutModulesRequiringLuminosityBlockSynchronization() const {
    cms::Exception ex("ModulesSynchingOnLumis");
    ex << "The framework is configured to use at least two streams, but the following modules\n"
       << "require synchronizing on LuminosityBlock boundaries:";
    bool found = false;
    for (auto worker : schedule_->allWorkers()) {
      if (worker->wantsGlobalLuminosityBlocks() and worker->globalLuminosityBlocksQueue()) {
        found = true;
        ex << "\n  " << worker->description()->moduleName() << " " << worker->description()->moduleLabel();
      }
    }
    if (found) {
      ex << "\n\nThe situation can be fixed by either\n"
         << " * modifying the modules to support concurrent LuminosityBlocks (preferred), or\n"
         << " * setting 'process.options.numberOfConcurrentLuminosityBlocks = 1' in the configuration file";
      throw ex;
    }
  }

  void EventProcessor::warnAboutLegacyModules() const {
    std::unique_ptr<LogSystem> s;
    for (auto worker : schedule_->allWorkers()) {
      if (worker->moduleConcurrencyType() == Worker::kLegacy) {
        if (not s) {
          s = std::make_unique<LogSystem>("LegacyModules");
          (*s) << "The following legacy modules are configured. Support for legacy modules\n"
                  "is going to end soon. These modules need to be converted to have type\n"
                  "edm::global, edm::stream, edm::one, or in rare cases edm::limited.";
        }
        (*s) << "\n  " << worker->description()->moduleName() << " " << worker->description()->moduleLabel();
      }
    }
  }
}  // namespace edm
