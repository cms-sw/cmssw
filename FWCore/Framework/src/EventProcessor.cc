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
#include "FWCore/Framework/interface/makeModuleTypeResolverMaker.h"
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
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/SendSourceTerminationSignalIfException.h"

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
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
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
#include "RunProcessingStatus.h"

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
  class PauseQueueSentry {
  public:
    PauseQueueSentry(edm::SerialTaskQueue& queue) : queue_(queue) { queue_.pause(); }
    ~PauseQueueSentry() { queue_.resume(); }

  private:
    edm::SerialTaskQueue& queue_;
  };
}  // namespace

namespace edm {

  namespace chain = waiting_task::chain;

  // ---------------------------------------------------------------
  std::unique_ptr<InputSource> makeInput(unsigned int moduleIndex,
                                         ParameterSet& params,
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
                         moduleIndex);

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
  std::shared_ptr<EDLooperBase> fillLooper(eventsetup::EventSetupsController& esController,
                                           eventsetup::EventSetupProvider& cp,
                                           ParameterSet& params,
                                           std::vector<std::string> const& loopers) {
    std::shared_ptr<EDLooperBase> vLooper;

    assert(1 == loopers.size());

    for (auto const& looperName : loopers) {
      ParameterSet* providerPSet = params.getPSetForUpdate(looperName);
      // Unlikely we would ever need the ModuleTypeResolver in Looper
      vLooper = eventsetup::LooperFactory::get()->addTo(esController, cp, *providerPSet, nullptr);
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
        moduleTypeResolverMaker_(makeModuleTypeResolverMaker(*parameterSet)),
        espController_(std::make_unique<eventsetup::EventSetupsController>(moduleTypeResolverMaker_.get())),
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
        exceptionMessageRuns_(false),
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
        moduleTypeResolverMaker_(makeModuleTypeResolverMaker(*parameterSet)),
        espController_(std::make_unique<eventsetup::EventSetupsController>(moduleTypeResolverMaker_.get())),
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
        exceptionMessageRuns_(false),
        exceptionMessageLumis_(false),
        forceLooperToEnd_(false),
        looperBeginJobRun_(false),
        forceESCacheClearOnNewRun_(false),
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
        moduleTypeResolverMaker_(makeModuleTypeResolverMaker(*processDesc->getProcessPSet())),
        espController_(std::make_unique<eventsetup::EventSetupsController>(moduleTypeResolverMaker_.get())),
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
        exceptionMessageRuns_(false),
        exceptionMessageLumis_(false),
        forceLooperToEnd_(false),
        looperBeginJobRun_(false),
        forceESCacheClearOnNewRun_(false),
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
    unsigned int nConcurrentLumis =
        optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentLuminosityBlocks");
    if (nConcurrentLumis == 0) {
      nConcurrentLumis = 2;
    }
    if (nConcurrentLumis > nStreams) {
      nConcurrentLumis = nStreams;
    }
    unsigned int nConcurrentRuns = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentRuns");
    if (nConcurrentRuns == 0 || nConcurrentRuns > nConcurrentLumis) {
      nConcurrentRuns = nConcurrentLumis;
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
    // This maximum simplifies to being equal nConcurrentLumis if nConcurrentRuns is 1.
    // Considering endRun, beginRun, and beginLumi we might need 3 concurrent IOVs per
    // concurrent run past the first in use cases where IOVs change within a run.
    unsigned int maxConcurrentIOVs =
        3 * nConcurrentRuns - 2 + ((nConcurrentLumis > nConcurrentRuns) ? (nConcurrentLumis - nConcurrentRuns) : 0);

    IllegalParameters::setThrowAnException(optionsPset.getUntrackedParameter<bool>("throwIfIllegalParameter"));

    printDependencies_ = optionsPset.getUntrackedParameter<bool>("printDependencies");
    deleteNonConsumedUnscheduledModules_ =
        optionsPset.getUntrackedParameter<bool>("deleteNonConsumedUnscheduledModules");
    //for now, if have a subProcess, don't allow early delete
    // In the future we should use the SubProcess's 'keep list' to decide what can be kept
    if (not hasSubProcesses) {
      branchesToDeleteEarly_ = optionsPset.getUntrackedParameter<std::vector<std::string>>("canDeleteEarly");
    }
    if (not branchesToDeleteEarly_.empty()) {
      auto referencePSets =
          optionsPset.getUntrackedParameter<std::vector<edm::ParameterSet>>("holdsReferencesToDeleteEarly");
      for (auto const& pset : referencePSets) {
        auto product = pset.getParameter<std::string>("product");
        auto references = pset.getParameter<std::vector<std::string>>("references");
        for (auto const& ref : references) {
          referencesToBranches_.emplace(product, ref);
        }
      }
      modulesToIgnoreForDeleteEarly_ =
          optionsPset.getUntrackedParameter<std::vector<std::string>>("modulesToIgnoreForDeleteEarly");
    }

    // Now do general initialization
    ScheduleItems items;

    //initialize the services
    auto& serviceSets = processDesc->getServicesPSets();
    ServiceToken token = items.initServices(serviceSets, *parameterSet, iToken, iLegacy, true);
    serviceToken_ = items.addCPRandTNS(*parameterSet, token);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    CMS_SA_ALLOW try {
      if (nThreads > 1) {
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

      runQueue_ = std::make_unique<LimitedTaskQueue>(nConcurrentRuns);
      lumiQueue_ = std::make_unique<LimitedTaskQueue>(nConcurrentLumis);
      streamQueues_.resize(nStreams);
      streamRunStatus_.resize(nStreams);
      streamLumiStatus_.resize(nStreams);

      processBlockHelper_ = std::make_shared<ProcessBlockHelper>();

      {
        std::optional<ScheduleItems::MadeModules> madeModules;

        //setup input and modules concurrently
        tbb::task_group group;

        // initialize the input source
        auto tempReg = std::make_shared<ProductRegistry>();
        auto sourceID = ModuleDescription::getUniqueID();

        group.run([&, this]() {
          // initialize the Schedule
          ServiceRegistry::Operate operate(serviceToken_);
          auto const& tns = ServiceRegistry::instance().get<service::TriggerNamesService>();
          madeModules =
              items.initModules(*parameterSet, tns, preallocations_, &processContext_, moduleTypeResolverMaker_.get());
        });

        group.run([&, this, tempReg]() {
          ServiceRegistry::Operate operate(serviceToken_);
          input_ = makeInput(sourceID,
                             *parameterSet,
                             *common,
                             /*items.preg(),*/ tempReg,
                             items.branchIDListHelper(),
                             get_underlying_safe(processBlockHelper_),
                             items.thinnedAssociationsHelper(),
                             items.actReg_,
                             items.processConfiguration(),
                             preallocations_);
        });

        group.wait();
        items.preg()->addFromInput(*tempReg);
        input_->switchTo(items.preg());

        {
          auto const& tns = ServiceRegistry::instance().get<service::TriggerNamesService>();
          schedule_ = items.finishSchedule(std::move(*madeModules),
                                           *parameterSet,
                                           tns,
                                           hasSubProcesses,
                                           preallocations_,
                                           &processContext_,
                                           *processBlockHelper_);
        }
      }

      // set the data members
      act_table_ = std::move(items.act_table_);
      actReg_ = items.actReg_;
      preg_ = items.preg();
      mergeableRunProductProcesses_.setProcessesWithMergeableRunProducts(*preg_);
      branchIDListHelper_ = items.branchIDListHelper();
      thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
      processConfiguration_ = items.processConfiguration();
      processContext_.setProcessConfiguration(processConfiguration_.get());

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

      for (unsigned int index = 0; index < preallocations_.numberOfRuns(); ++index) {
        auto rp = std::make_unique<RunPrincipal>(
            preg(), *processConfiguration_, historyAppender_.get(), index, true, &mergeableRunProductProcesses_);
        principalCache_.insert(std::move(rp));
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
                                   &processContext_,
                                   moduleTypeResolverMaker_);
      }
    } catch (...) {
      //in case of an exception, make sure Services are available
      // during the following destructors
      espController_ = nullptr;
      esp_ = nullptr;
      schedule_ = nullptr;
      input_ = nullptr;
      looper_ = nullptr;
      actReg_ = nullptr;
      throw;
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
    edm::FinalWaitingTask task{taskGroup_};
    espController_->endIOVsAsync(edm::WaitingTaskHolder{taskGroup_, &task});
    task.waitNoThrow();
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
      auto modulesToSkip = std::move(modulesToIgnoreForDeleteEarly_);
      auto branchesToDeleteEarly = std::move(branchesToDeleteEarly_);
      auto referencesToBranches = std::move(referencesToBranches_);
      schedule_->initializeEarlyDelete(branchesToDeleteEarly, referencesToBranches, modulesToSkip, *preg_);
    }

    actReg_->preBeginJobSignal_(pathsAndConsumesOfModules_, processContext_);

    if (preallocations_.numberOfLuminosityBlocks() > 1) {
      throwAboutModulesRequiringLuminosityBlockSynchronization();
    }
    if (preallocations_.numberOfRuns() > 1) {
      warnAboutModulesRequiringRunSynchronization();
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

    oneapi::tbb::task_group group;
    FinalWaitingTask last{group};
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
    last.wait();
  }

  void EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c(
        "Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    using namespace edm::waiting_task::chain;

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};

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
    waitTask.waitNoThrow();

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

  EventProcessor::StatusCode EventProcessor::runToCompletion() {
    beginJob();  //make sure this was called

    // make the services available
    ServiceRegistry::Operate operate(serviceToken_);

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
      if (exceptionMessageRuns_) {
        std::string message(
            "Another exception was caught while trying to clean up runs after the primary fatal exception.");
        e.addAdditionalInfo(message);
        if (e.alreadyPrinted()) {
          LogAbsolute("Additional Exceptions") << message;
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
    return epSuccess;
  }

  void EventProcessor::readFile() {
    FDEBUG(1) << " \treadFile\n";
    size_t size = preg_->size();
    SendSourceTerminationSignalIfException sentry(actReg_.get());

    if (streamRunActive_ > 0) {
      streamRunStatus_[0]->runPrincipal()->preReadFile();
      streamRunStatus_[0]->runPrincipal()->adjustIndexesAfterProductRegistryAddition();
    }

    if (streamLumiActive_ > 0) {
      streamLumiStatus_[0]->lumiPrincipal()->adjustIndexesAfterProductRegistryAddition();
    }

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
    FinalWaitingTask globalWaitTask{taskGroup_};

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    beginGlobalTransitionAsync<Traits>(
        WaitingTaskHolder(taskGroup_, &globalWaitTask), *schedule_, transitionInfo, serviceToken_, subProcesses_);

    globalWaitTask.wait();
    beginProcessBlockSucceeded = true;
  }

  void EventProcessor::inputProcessBlocks() {
    input_->fillProcessBlockHelper();
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
    while (input_->nextProcessBlock(processBlockPrincipal)) {
      readProcessBlock(processBlockPrincipal);

      using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>;
      FinalWaitingTask globalWaitTask{taskGroup_};

      ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
      beginGlobalTransitionAsync<Traits>(
          WaitingTaskHolder(taskGroup_, &globalWaitTask), *schedule_, transitionInfo, serviceToken_, subProcesses_);

      globalWaitTask.wait();

      FinalWaitingTask writeWaitTask{taskGroup_};
      writeProcessBlockAsync(edm::WaitingTaskHolder{taskGroup_, &writeWaitTask}, ProcessBlockType::Input);
      writeWaitTask.wait();

      processBlockPrincipal.clearPrincipal();
      for (auto& s : subProcesses_) {
        s.clearProcessBlockPrincipal(ProcessBlockType::Input);
      }
    }
  }

  void EventProcessor::endProcessBlock(bool cleaningUpAfterException, bool beginProcessBlockSucceeded) {
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();

    using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
    FinalWaitingTask globalWaitTask{taskGroup_};

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    endGlobalTransitionAsync<Traits>(WaitingTaskHolder(taskGroup_, &globalWaitTask),
                                     *schedule_,
                                     transitionInfo,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
    globalWaitTask.wait();

    if (beginProcessBlockSucceeded) {
      FinalWaitingTask writeWaitTask{taskGroup_};
      writeProcessBlockAsync(edm::WaitingTaskHolder{taskGroup_, &writeWaitTask}, ProcessBlockType::New);
      writeWaitTask.wait();
    }

    processBlockPrincipal.clearPrincipal();
    for (auto& s : subProcesses_) {
      s.clearProcessBlockPrincipal(ProcessBlockType::New);
    }
  }

  InputSource::ItemType EventProcessor::processRuns() {
    FinalWaitingTask waitTask{taskGroup_};
    assert(lastTransitionType() == InputSource::IsRun);
    if (streamRunActive_ == 0) {
      assert(streamLumiActive_ == 0);

      beginRunAsync(IOVSyncValue(EventID(input_->run(), 0, 0), input_->runAuxiliary()->beginTime()),
                    WaitingTaskHolder{taskGroup_, &waitTask});
    } else {
      assert(streamRunActive_ == preallocations_.numberOfStreams());

      auto runStatus = streamRunStatus_[0];

      while (lastTransitionType() == InputSource::IsRun and runStatus->runPrincipal()->run() == input_->run() and
             runStatus->runPrincipal()->reducedProcessHistoryID() == input_->reducedProcessHistoryID()) {
        readAndMergeRun(*runStatus);
        nextTransitionType();
      }

      WaitingTaskHolder holder{taskGroup_, &waitTask};
      runStatus->setHolderOfTaskInProcessRuns(holder);
      if (streamLumiActive_ > 0) {
        assert(streamLumiActive_ == preallocations_.numberOfStreams());
        continueLumiAsync(std::move(holder));
      } else {
        handleNextItemAfterMergingRunEntries(std::move(runStatus), std::move(holder));
      }
    }
    waitTask.wait();
    return lastTransitionType();
  }

  void EventProcessor::beginRunAsync(IOVSyncValue const& iSync, WaitingTaskHolder iHolder) {
    if (iHolder.taskHasFailed()) {
      return;
    }

    actReg_->esSyncIOVQueuingSignal_.emit(iSync);

    auto status = std::make_shared<RunProcessingStatus>(preallocations_.numberOfStreams(), iHolder);

    chain::first([this, &status, &iSync](auto nextTask) {
      espController_->runOrQueueEventSetupForInstanceAsync(iSync,
                                                           nextTask,
                                                           status->endIOVWaitingTasks(),
                                                           status->eventSetupImpls(),
                                                           queueWhichWaitsForIOVsToFinish_,
                                                           actReg_.get(),
                                                           serviceToken_,
                                                           forceESCacheClearOnNewRun_);
    }) | chain::then([this, status, iSync](std::exception_ptr const* iException, auto nextTask) {
      CMS_SA_ALLOW try {
        if (iException) {
          WaitingTaskHolder copyHolder(nextTask);
          copyHolder.doneWaiting(*iException);
          // Finish handling the exception in the task pushed to runQueue_
        }
        ServiceRegistry::Operate operate(serviceToken_);
        actReg_->postESSyncIOVSignal_.emit(iSync);

        runQueue_->pushAndPause(
            *nextTask.group(),
            [this, postRunQueueTask = nextTask, status](edm::LimitedTaskQueue::Resumer iResumer) mutable {
              CMS_SA_ALLOW try {
                if (postRunQueueTask.taskHasFailed()) {
                  status->resetBeginResources();
                  queueWhichWaitsForIOVsToFinish_.resume();
                  return;
                }

                status->setResumer(std::move(iResumer));

                sourceResourcesAcquirer_.serialQueueChain().push(
                    *postRunQueueTask.group(), [this, postSourceTask = postRunQueueTask, status]() mutable {
                      CMS_SA_ALLOW try {
                        ServiceRegistry::Operate operate(serviceToken_);

                        if (postSourceTask.taskHasFailed()) {
                          status->resetBeginResources();
                          queueWhichWaitsForIOVsToFinish_.resume();
                          status->resumeGlobalRunQueue();
                          return;
                        }

                        status->setRunPrincipal(readRun());

                        RunPrincipal& runPrincipal = *status->runPrincipal();
                        {
                          SendSourceTerminationSignalIfException sentry(actReg_.get());
                          input_->doBeginRun(runPrincipal, &processContext_);
                          sentry.completedSuccessfully();
                        }

                        EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());
                        if (looper_ && looperBeginJobRun_ == false) {
                          looper_->copyInfo(ScheduleInfo(schedule_.get()));

                          oneapi::tbb::task_group group;
                          FinalWaitingTask waitTask{group};
                          using namespace edm::waiting_task::chain;
                          chain::first([this, &es](auto nextTask) {
                            looper_->esPrefetchAsync(nextTask, es, Transition::BeginRun, serviceToken_);
                          }) | then([this, &es](auto nextTask) mutable {
                            looper_->beginOfJob(es);
                            looperBeginJobRun_ = true;
                            looper_->doStartingNewLoop();
                          }) | runLast(WaitingTaskHolder(group, &waitTask));
                          waitTask.wait();
                        }

                        using namespace edm::waiting_task::chain;
                        chain::first([this, status](auto nextTask) mutable {
                          CMS_SA_ALLOW try { readAndMergeRunEntriesAsync(std::move(status), nextTask); } catch (...) {
                            status->setStopBeforeProcessingRun(true);
                            nextTask.doneWaiting(std::current_exception());
                          }
                        }) | then([this, status, &es](auto nextTask) {
                          if (status->stopBeforeProcessingRun()) {
                            return;
                          }
                          RunTransitionInfo transitionInfo(*status->runPrincipal(), es, &status->eventSetupImpls());
                          using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>;
                          beginGlobalTransitionAsync<Traits>(
                              nextTask, *schedule_, transitionInfo, serviceToken_, subProcesses_);
                        }) | then([status](auto nextTask) mutable {
                          if (status->stopBeforeProcessingRun()) {
                            return;
                          }
                          status->globalBeginDidSucceed();
                        }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                          if (status->stopBeforeProcessingRun()) {
                            return;
                          }
                          looper_->prefetchAsync(
                              nextTask, serviceToken_, Transition::BeginRun, *status->runPrincipal(), es);
                        }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                          if (status->stopBeforeProcessingRun()) {
                            return;
                          }
                          ServiceRegistry::Operate operateLooper(serviceToken_);
                          looper_->doBeginRun(*status->runPrincipal(), es, &processContext_);
                        }) | then([this, status](std::exception_ptr const* iException, auto holder) mutable {
                          bool precedingTasksSucceeded = true;
                          if (iException) {
                            precedingTasksSucceeded = false;
                            WaitingTaskHolder copyHolder(holder);
                            copyHolder.doneWaiting(*iException);
                          }

                          if (status->stopBeforeProcessingRun()) {
                            // We just quit now if there was a failure when merging runs
                            status->resetBeginResources();
                            queueWhichWaitsForIOVsToFinish_.resume();
                            status->resumeGlobalRunQueue();
                            return;
                          }
                          CMS_SA_ALLOW try {
                            // Under normal circumstances, this task runs after endRun has completed for all streams
                            // and global endLumi has completed for all lumis contained in this run
                            auto globalEndRunTask =
                                edm::make_waiting_task([this, status](std::exception_ptr const*) mutable {
                                  WaitingTaskHolder taskHolder = status->holderOfTaskInProcessRuns();
                                  status->holderOfTaskInProcessRuns().doneWaiting(std::exception_ptr{});
                                  globalEndRunAsync(std::move(taskHolder), std::move(status));
                                });
                            status->setGlobalEndRunHolder(WaitingTaskHolder{*holder.group(), globalEndRunTask});
                          } catch (...) {
                            status->resetBeginResources();
                            queueWhichWaitsForIOVsToFinish_.resume();
                            status->resumeGlobalRunQueue();
                            holder.doneWaiting(std::current_exception());
                            return;
                          }

                          // After this point we are committed to end the run via endRunAsync

                          ServiceRegistry::Operate operate(serviceToken_);

                          // The only purpose of the pause is to cause stream begin run to execute before
                          // global begin lumi in the single threaded case (maintains consistency with
                          // the order that existed before concurrent runs were implemented).
                          PauseQueueSentry pauseQueueSentry(streamQueuesInserter_);

                          CMS_SA_ALLOW try {
                            streamQueuesInserter_.push(
                                *holder.group(), [this, status, precedingTasksSucceeded, holder]() mutable {
                                  for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
                                    CMS_SA_ALLOW try {
                                      streamQueues_[i].push(
                                          *holder.group(),
                                          [this, i, status, precedingTasksSucceeded, holder]() mutable {
                                            streamBeginRunAsync(
                                                i, std::move(status), precedingTasksSucceeded, std::move(holder));
                                          });
                                    } catch (...) {
                                      if (status->streamFinishedBeginRun()) {
                                        WaitingTaskHolder copyHolder(holder);
                                        copyHolder.doneWaiting(std::current_exception());
                                        status->resetBeginResources();
                                        queueWhichWaitsForIOVsToFinish_.resume();
                                        exceptionRunStatus_ = status;
                                      }
                                    }
                                  }
                                });
                          } catch (...) {
                            WaitingTaskHolder copyHolder(holder);
                            copyHolder.doneWaiting(std::current_exception());
                            status->resetBeginResources();
                            queueWhichWaitsForIOVsToFinish_.resume();
                            exceptionRunStatus_ = status;
                          }
                          handleNextItemAfterMergingRunEntries(status, holder);
                        }) | runLast(postSourceTask);
                      } catch (...) {
                        status->resetBeginResources();
                        queueWhichWaitsForIOVsToFinish_.resume();
                        status->resumeGlobalRunQueue();
                        postSourceTask.doneWaiting(std::current_exception());
                      }
                    });  // task in sourceResourcesAcquirer
              } catch (...) {
                status->resetBeginResources();
                queueWhichWaitsForIOVsToFinish_.resume();
                status->resumeGlobalRunQueue();
                postRunQueueTask.doneWaiting(std::current_exception());
              }
            });  // task in runQueue
      } catch (...) {
        status->resetBeginResources();
        queueWhichWaitsForIOVsToFinish_.resume();
        nextTask.doneWaiting(std::current_exception());
      }
    }) | chain::runLast(std::move(iHolder));
  }

  void EventProcessor::streamBeginRunAsync(unsigned int iStream,
                                           std::shared_ptr<RunProcessingStatus> status,
                                           bool precedingTasksSucceeded,
                                           WaitingTaskHolder iHolder) {
    // These shouldn't throw
    streamQueues_[iStream].pause();
    ++streamRunActive_;
    streamRunStatus_[iStream] = std::move(status);

    CMS_SA_ALLOW try {
      using namespace edm::waiting_task::chain;
      chain::first([this, iStream, precedingTasksSucceeded](auto nextTask) {
        if (precedingTasksSucceeded) {
          RunProcessingStatus& rs = *streamRunStatus_[iStream];
          RunTransitionInfo transitionInfo(
              *rs.runPrincipal(), rs.eventSetupImpl(esp_->subProcessIndex()), &rs.eventSetupImpls());
          using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>;
          beginStreamTransitionAsync<Traits>(
              std::move(nextTask), *schedule_, iStream, transitionInfo, serviceToken_, subProcesses_);
        }
      }) | then([this, iStream](std::exception_ptr const* exceptionFromBeginStreamRun, auto nextTask) {
        if (exceptionFromBeginStreamRun) {
          nextTask.doneWaiting(*exceptionFromBeginStreamRun);
        }
        releaseBeginRunResources(iStream);
      }) | runLast(iHolder);
    } catch (...) {
      releaseBeginRunResources(iStream);
      iHolder.doneWaiting(std::current_exception());
    }
  }

  void EventProcessor::releaseBeginRunResources(unsigned int iStream) {
    auto& status = streamRunStatus_[iStream];
    if (status->streamFinishedBeginRun()) {
      status->resetBeginResources();
      queueWhichWaitsForIOVsToFinish_.resume();
    }
    streamQueues_[iStream].resume();
  }

  void EventProcessor::endRunAsync(std::shared_ptr<RunProcessingStatus> iRunStatus, WaitingTaskHolder iHolder) {
    RunPrincipal& runPrincipal = *iRunStatus->runPrincipal();
    iRunStatus->setEndTime();
    IOVSyncValue ts(
        EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
        runPrincipal.endTime());
    CMS_SA_ALLOW try { actReg_->esSyncIOVQueuingSignal_.emit(ts); } catch (...) {
      WaitingTaskHolder copyHolder(iHolder);
      copyHolder.doneWaiting(std::current_exception());
    }

    chain::first([this, &iRunStatus, &ts](auto nextTask) {
      espController_->runOrQueueEventSetupForInstanceAsync(ts,
                                                           nextTask,
                                                           iRunStatus->endIOVWaitingTasksEndRun(),
                                                           iRunStatus->eventSetupImplsEndRun(),
                                                           queueWhichWaitsForIOVsToFinish_,
                                                           actReg_.get(),
                                                           serviceToken_);
    }) | chain::then([this, iRunStatus, ts](std::exception_ptr const* iException, auto nextTask) {
      if (iException) {
        iRunStatus->setEndingEventSetupSucceeded(false);
        handleEndRunExceptions(*iException, nextTask);
      }
      ServiceRegistry::Operate operate(serviceToken_);
      CMS_SA_ALLOW try { actReg_->postESSyncIOVSignal_.emit(ts); } catch (...) {
        WaitingTaskHolder copyHolder(nextTask);
        copyHolder.doneWaiting(std::current_exception());
      }

      streamQueuesInserter_.push(*nextTask.group(), [this, nextTask]() mutable {
        for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
          CMS_SA_ALLOW try {
            streamQueues_[i].push(*nextTask.group(), [this, i, nextTask]() mutable {
              streamQueues_[i].pause();
              streamEndRunAsync(std::move(nextTask), i);
            });
          } catch (...) {
            WaitingTaskHolder copyHolder(nextTask);
            copyHolder.doneWaiting(std::current_exception());
          }
        }
      });

      if (lastTransitionType() == InputSource::IsRun) {
        CMS_SA_ALLOW try {
          beginRunAsync(IOVSyncValue(EventID(input_->run(), 0, 0), input_->runAuxiliary()->beginTime()), nextTask);
        } catch (...) {
          WaitingTaskHolder copyHolder(nextTask);
          copyHolder.doneWaiting(std::current_exception());
        }
      }
    }) | chain::runLast(std::move(iHolder));
  }

  void EventProcessor::handleEndRunExceptions(std::exception_ptr iException, WaitingTaskHolder const& holder) {
    if (holder.taskHasFailed()) {
      setExceptionMessageRuns();
    } else {
      WaitingTaskHolder tmp(holder);
      tmp.doneWaiting(iException);
    }
  }

  void EventProcessor::globalEndRunAsync(WaitingTaskHolder iTask, std::shared_ptr<RunProcessingStatus> iRunStatus) {
    auto& runPrincipal = *(iRunStatus->runPrincipal());
    bool didGlobalBeginSucceed = iRunStatus->didGlobalBeginSucceed();
    bool cleaningUpAfterException = iRunStatus->cleaningUpAfterException() || iTask.taskHasFailed();
    EventSetupImpl const& es = iRunStatus->eventSetupImplEndRun(esp_->subProcessIndex());
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = &iRunStatus->eventSetupImplsEndRun();
    bool endingEventSetupSucceeded = iRunStatus->endingEventSetupSucceeded();

    MergeableRunProductMetadata* mergeableRunProductMetadata = runPrincipal.mergeableRunProductMetadata();
    using namespace edm::waiting_task::chain;
    chain::first([this, &runPrincipal, &es, &eventSetupImpls, cleaningUpAfterException, endingEventSetupSucceeded](
                     auto nextTask) {
      if (endingEventSetupSucceeded) {
        RunTransitionInfo transitionInfo(runPrincipal, es, eventSetupImpls);
        using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>;
        endGlobalTransitionAsync<Traits>(
            std::move(nextTask), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
      }
    }) |
        ifThen(looper_ && endingEventSetupSucceeded,
               [this, &runPrincipal, &es](auto nextTask) {
                 looper_->prefetchAsync(std::move(nextTask), serviceToken_, Transition::EndRun, runPrincipal, es);
               }) |
        ifThen(looper_ && endingEventSetupSucceeded,
               [this, &runPrincipal, &es](auto nextTask) {
                 ServiceRegistry::Operate operate(serviceToken_);
                 looper_->doEndRun(runPrincipal, es, &processContext_);
               }) |
        ifThen(didGlobalBeginSucceed && endingEventSetupSucceeded,
               [this, mergeableRunProductMetadata, &runPrincipal = runPrincipal](auto nextTask) {
                 mergeableRunProductMetadata->preWriteRun();
                 writeRunAsync(nextTask, runPrincipal, mergeableRunProductMetadata);
               }) |
        then([status = std::move(iRunStatus),
              this,
              didGlobalBeginSucceed,
              mergeableRunProductMetadata,
              endingEventSetupSucceeded](std::exception_ptr const* iException, auto nextTask) mutable {
          if (didGlobalBeginSucceed && endingEventSetupSucceeded) {
            mergeableRunProductMetadata->postWriteRun();
          }
          if (iException) {
            handleEndRunExceptions(*iException, nextTask);
          }
          ServiceRegistry::Operate operate(serviceToken_);

          std::exception_ptr ptr;

          // Try hard to clean up resources so the
          // process can terminate in a controlled
          // fashion even after exceptions have occurred.
          CMS_SA_ALLOW try { clearRunPrincipal(*status); } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }
          CMS_SA_ALLOW try {
            status->resumeGlobalRunQueue();
            queueWhichWaitsForIOVsToFinish_.resume();
          } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }
          CMS_SA_ALLOW try {
            status->resetEndResources();
            status.reset();
          } catch (...) {
            if (not ptr) {
              ptr = std::current_exception();
            }
          }

          if (ptr && !iException) {
            handleEndRunExceptions(ptr, nextTask);
          }
        }) |
        runLast(std::move(iTask));
  }

  void EventProcessor::streamEndRunAsync(WaitingTaskHolder iTask, unsigned int iStreamIndex) {
    CMS_SA_ALLOW try {
      if (!streamRunStatus_[iStreamIndex]) {
        if (exceptionRunStatus_->streamFinishedRun()) {
          exceptionRunStatus_->globalEndRunHolder().doneWaiting(std::exception_ptr());
          exceptionRunStatus_.reset();
        }
        return;
      }

      auto runDoneTask =
          edm::make_waiting_task([this, iTask, iStreamIndex](std::exception_ptr const* iException) mutable {
            if (iException) {
              handleEndRunExceptions(*iException, iTask);
            }

            auto runStatus = streamRunStatus_[iStreamIndex];

            //reset status before releasing queue else get race condition
            if (runStatus->streamFinishedRun()) {
              runStatus->globalEndRunHolder().doneWaiting(std::exception_ptr());
            }
            streamRunStatus_[iStreamIndex].reset();
            --streamRunActive_;
            streamQueues_[iStreamIndex].resume();
          });

      WaitingTaskHolder runDoneTaskHolder{*iTask.group(), runDoneTask};

      auto runStatus = streamRunStatus_[iStreamIndex].get();

      if (runStatus->didGlobalBeginSucceed() && runStatus->endingEventSetupSucceeded()) {
        EventSetupImpl const& es = runStatus->eventSetupImplEndRun(esp_->subProcessIndex());
        auto eventSetupImpls = &runStatus->eventSetupImplsEndRun();
        bool cleaningUpAfterException = runStatus->cleaningUpAfterException() || iTask.taskHasFailed();

        auto& runPrincipal = *runStatus->runPrincipal();
        using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>;
        RunTransitionInfo transitionInfo(runPrincipal, es, eventSetupImpls);
        endStreamTransitionAsync<Traits>(std::move(runDoneTaskHolder),
                                         *schedule_,
                                         iStreamIndex,
                                         transitionInfo,
                                         serviceToken_,
                                         subProcesses_,
                                         cleaningUpAfterException);
      }
    } catch (...) {
      handleEndRunExceptions(std::current_exception(), iTask);
    }
  }

  void EventProcessor::endUnfinishedRun(bool cleaningUpAfterException) {
    if (streamRunActive_ > 0) {
      FinalWaitingTask waitTask{taskGroup_};

      auto runStatus = streamRunStatus_[0].get();
      runStatus->setCleaningUpAfterException(cleaningUpAfterException);
      WaitingTaskHolder holder{taskGroup_, &waitTask};
      runStatus->setHolderOfTaskInProcessRuns(holder);
      lastSourceTransition_ = InputSource::IsStop;
      endRunAsync(streamRunStatus_[0], std::move(holder));
      waitTask.wait();
    }
  }

  void EventProcessor::beginLumiAsync(IOVSyncValue const& iSync,
                                      std::shared_ptr<RunProcessingStatus> iRunStatus,
                                      edm::WaitingTaskHolder iHolder) {
    actReg_->esSyncIOVQueuingSignal_.emit(iSync);

    auto status = std::make_shared<LuminosityBlockProcessingStatus>(preallocations_.numberOfStreams());
    chain::first([this, &iSync, &status](auto nextTask) {
      espController_->runOrQueueEventSetupForInstanceAsync(iSync,
                                                           nextTask,
                                                           status->endIOVWaitingTasks(),
                                                           status->eventSetupImpls(),
                                                           queueWhichWaitsForIOVsToFinish_,
                                                           actReg_.get(),
                                                           serviceToken_);
    }) | chain::then([this, status, iRunStatus, iSync](std::exception_ptr const* iException, auto nextTask) {
      CMS_SA_ALLOW try {
        //the call to doneWaiting will cause the count to decrement
        if (iException) {
          WaitingTaskHolder copyHolder(nextTask);
          copyHolder.doneWaiting(*iException);
        }

        ServiceRegistry::Operate operate(serviceToken_);
        actReg_->postESSyncIOVSignal_.emit(iSync);

        lumiQueue_->pushAndPause(
            *nextTask.group(),
            [this, postLumiQueueTask = nextTask, status, iRunStatus](edm::LimitedTaskQueue::Resumer iResumer) mutable {
              CMS_SA_ALLOW try {
                if (postLumiQueueTask.taskHasFailed()) {
                  status->resetResources();
                  queueWhichWaitsForIOVsToFinish_.resume();
                  endRunAsync(iRunStatus, postLumiQueueTask);
                  return;
                }

                status->setResumer(std::move(iResumer));

                sourceResourcesAcquirer_.serialQueueChain().push(
                    *postLumiQueueTask.group(),
                    [this, postSourceTask = postLumiQueueTask, status, iRunStatus]() mutable {
                      CMS_SA_ALLOW try {
                        ServiceRegistry::Operate operate(serviceToken_);

                        if (postSourceTask.taskHasFailed()) {
                          status->resetResources();
                          queueWhichWaitsForIOVsToFinish_.resume();
                          endRunAsync(iRunStatus, postSourceTask);
                          return;
                        }

                        status->setLumiPrincipal(readLuminosityBlock(iRunStatus->runPrincipal()));

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
                        chain::first([this, status](auto nextTask) mutable {
                          readAndMergeLumiEntriesAsync(std::move(status), std::move(nextTask));
                          firstItemAfterLumiMerge_ = true;
                        }) | then([this, status, &es, &lumiPrincipal](auto nextTask) {
                          LumiTransitionInfo transitionInfo(lumiPrincipal, es, &status->eventSetupImpls());
                          using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>;
                          beginGlobalTransitionAsync<Traits>(
                              nextTask, *schedule_, transitionInfo, serviceToken_, subProcesses_);
                        }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                          looper_->prefetchAsync(
                              nextTask, serviceToken_, Transition::BeginLuminosityBlock, *(status->lumiPrincipal()), es);
                        }) | ifThen(looper_, [this, status, &es](auto nextTask) {
                          status->globalBeginDidSucceed();
                          ServiceRegistry::Operate operateLooper(serviceToken_);
                          looper_->doBeginLuminosityBlock(*(status->lumiPrincipal()), es, &processContext_);
                        }) | then([this, status, iRunStatus](std::exception_ptr const* iException, auto holder) mutable {
                          if (iException) {
                            status->resetResources();
                            queueWhichWaitsForIOVsToFinish_.resume();
                            WaitingTaskHolder copyHolder(holder);
                            copyHolder.doneWaiting(*iException);
                            endRunAsync(iRunStatus, holder);
                          } else {
                            if (not looper_) {
                              status->globalBeginDidSucceed();
                            }

                            status->setGlobalEndRunHolder(iRunStatus->globalEndRunHolder());

                            EventSetupImpl const& es = status->eventSetupImpl(esp_->subProcessIndex());
                            using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>;

                            streamQueuesInserter_.push(*holder.group(), [this, status, holder, &es]() mutable {
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
                                    beginStreamTransitionAsync<Traits>(std::move(nextTask),
                                                                       *schedule_,
                                                                       i,
                                                                       transitionInfo,
                                                                       serviceToken_,
                                                                       subProcesses_);
                                  }) |
                                      then([this, i](std::exception_ptr const* exceptionFromBeginStreamLumi,
                                                     auto nextTask) {
                                        if (exceptionFromBeginStreamLumi) {
                                          WaitingTaskHolder copyHolder(nextTask);
                                          copyHolder.doneWaiting(*exceptionFromBeginStreamLumi);
                                        }
                                        handleNextEventForStreamAsync(std::move(nextTask), i);
                                      }) |
                                      runLast(std::move(holder));
                                });
                              }  // end for loop over streams
                            });
                          }
                        }) | runLast(postSourceTask);
                      } catch (...) {
                        status->resetResources();
                        queueWhichWaitsForIOVsToFinish_.resume();
                        WaitingTaskHolder copyHolder(postSourceTask);
                        copyHolder.doneWaiting(std::current_exception());
                        endRunAsync(iRunStatus, postSourceTask);
                      }
                    });  // task in sourceResourcesAcquirer
              } catch (...) {
                status->resetResources();
                queueWhichWaitsForIOVsToFinish_.resume();
                WaitingTaskHolder copyHolder(postLumiQueueTask);
                copyHolder.doneWaiting(std::current_exception());
                endRunAsync(iRunStatus, postLumiQueueTask);
              }
            });  // task in lumiQueue
      } catch (...) {
        status->resetResources();
        queueWhichWaitsForIOVsToFinish_.resume();
        WaitingTaskHolder copyHolder(nextTask);
        copyHolder.doneWaiting(std::current_exception());
        endRunAsync(iRunStatus, nextTask);
      }
    }) | chain::runLast(std::move(iHolder));
  }

  void EventProcessor::continueLumiAsync(edm::WaitingTaskHolder iHolder) {
    chain::first([this](auto nextTask) {
      //all streams are sharing the same status at the moment
      auto status = streamLumiStatus_[0];  //read from streamLumiActive_ happened in calling routine
      status->setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kProcessing);

      while (lastTransitionType() == InputSource::IsLumi and
             status->lumiPrincipal()->luminosityBlock() == input_->luminosityBlock()) {
        readAndMergeLumi(*status);
        nextTransitionType();
      }
      firstItemAfterLumiMerge_ = true;
    }) | chain::then([this](auto nextTask) mutable {
      unsigned int streamIndex = 0;
      oneapi::tbb::task_arena arena{oneapi::tbb::task_arena::attach()};
      for (; streamIndex < preallocations_.numberOfStreams() - 1; ++streamIndex) {
        arena.enqueue([this, streamIndex, h = nextTask]() { handleNextEventForStreamAsync(h, streamIndex); });
      }
      nextTask.group()->run(
          [this, streamIndex, h = std::move(nextTask)]() { handleNextEventForStreamAsync(h, streamIndex); });
    }) | chain::runLast(std::move(iHolder));
  }

  void EventProcessor::handleEndLumiExceptions(std::exception_ptr iException, WaitingTaskHolder const& holder) {
    if (holder.taskHasFailed()) {
      setExceptionMessageLumis();
    } else {
      WaitingTaskHolder tmp(holder);
      tmp.doneWaiting(iException);
    }
  }

  void EventProcessor::globalEndLumiAsync(edm::WaitingTaskHolder iTask,
                                          std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus) {
    // Get some needed info out of the status object before moving
    // it into finalTaskForThisLumi.
    auto& lp = *(iLumiStatus->lumiPrincipal());
    bool didGlobalBeginSucceed = iLumiStatus->didGlobalBeginSucceed();
    bool cleaningUpAfterException = iLumiStatus->cleaningUpAfterException() || iTask.taskHasFailed();
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
    }) | then([status = std::move(iLumiStatus), this](std::exception_ptr const* iException, auto nextTask) mutable {
      if (iException) {
        handleEndLumiExceptions(*iException, nextTask);
      }
      ServiceRegistry::Operate operate(serviceToken_);

      std::exception_ptr ptr;

      // Try hard to clean up resources so the
      // process can terminate in a controlled
      // fashion even after exceptions have occurred.
      // Caught exception is passed to handleEndLumiExceptions()
      CMS_SA_ALLOW try { clearLumiPrincipal(*status); } catch (...) {
        if (not ptr) {
          ptr = std::current_exception();
        }
      }
      // Caught exception is passed to handleEndLumiExceptions()
      CMS_SA_ALLOW try { queueWhichWaitsForIOVsToFinish_.resume(); } catch (...) {
        if (not ptr) {
          ptr = std::current_exception();
        }
      }
      // Caught exception is passed to handleEndLumiExceptions()
      CMS_SA_ALLOW try {
        status->resetResources();
        status->globalEndRunHolderDoneWaiting();
        status.reset();
      } catch (...) {
        if (not ptr) {
          ptr = std::current_exception();
        }
      }

      if (ptr && !iException) {
        handleEndLumiExceptions(ptr, nextTask);
      }
    }) | runLast(std::move(iTask));
  }

  void EventProcessor::streamEndLumiAsync(edm::WaitingTaskHolder iTask, unsigned int iStreamIndex) {
    auto t = edm::make_waiting_task([this, iStreamIndex, iTask](std::exception_ptr const* iException) mutable {
      auto status = streamLumiStatus_[iStreamIndex];
      if (iException) {
        handleEndLumiExceptions(*iException, iTask);
      }

      // reset status before releasing queue else get race condition
      streamLumiStatus_[iStreamIndex].reset();
      --streamLumiActive_;
      streamQueues_[iStreamIndex].resume();

      //are we the last one?
      if (status->streamFinishedLumi()) {
        globalEndLumiAsync(iTask, std::move(status));
      }
    });

    edm::WaitingTaskHolder lumiDoneTask{*iTask.group(), t};

    // Need to be sure the lumi status is released before lumiDoneTask can every be called.
    // therefore we do not want to hold the shared_ptr
    auto lumiStatus = streamLumiStatus_[iStreamIndex].get();
    lumiStatus->setEndTime();

    EventSetupImpl const& es = lumiStatus->eventSetupImpl(esp_->subProcessIndex());
    auto eventSetupImpls = &lumiStatus->eventSetupImpls();
    bool cleaningUpAfterException = lumiStatus->cleaningUpAfterException() || iTask.taskHasFailed();

    if (lumiStatus->didGlobalBeginSucceed()) {
      auto& lumiPrincipal = *lumiStatus->lumiPrincipal();
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

  void EventProcessor::endUnfinishedLumi(bool cleaningUpAfterException) {
    if (streamRunActive_ == 0) {
      assert(streamLumiActive_ == 0);
    } else {
      assert(streamRunActive_ == preallocations_.numberOfStreams());
      if (streamLumiActive_ > 0) {
        FinalWaitingTask globalWaitTask{taskGroup_};
        assert(streamLumiActive_ == preallocations_.numberOfStreams());
        streamLumiStatus_[0]->setCleaningUpAfterException(cleaningUpAfterException);
        for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
          streamEndLumiAsync(WaitingTaskHolder{taskGroup_, &globalWaitTask}, i);
        }
        globalWaitTask.wait();
      }
    }
  }

  void EventProcessor::readProcessBlock(ProcessBlockPrincipal& processBlockPrincipal) {
    SendSourceTerminationSignalIfException sentry(actReg_.get());
    input_->readProcessBlock(processBlockPrincipal);
    sentry.completedSuccessfully();
  }

  std::shared_ptr<RunPrincipal> EventProcessor::readRun() {
    auto rp = principalCache_.getAvailableRunPrincipalPtr();
    assert(rp);
    rp->setAux(*input_->runAuxiliary());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readRun(*rp, *historyAppender_);
      sentry.completedSuccessfully();
    }
    assert(input_->reducedProcessHistoryID() == rp->reducedProcessHistoryID());
    return rp;
  }

  void EventProcessor::readAndMergeRun(RunProcessingStatus& iStatus) {
    RunPrincipal& runPrincipal = *iStatus.runPrincipal();

    bool runOK = runPrincipal.adjustToNewProductRegistry(*preg_);
    assert(runOK);
    runPrincipal.mergeAuxiliary(*input_->runAuxiliary());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readAndMergeRun(runPrincipal);
      sentry.completedSuccessfully();
    }
  }

  std::shared_ptr<LuminosityBlockPrincipal> EventProcessor::readLuminosityBlock(std::shared_ptr<RunPrincipal> rp) {
    auto lbp = principalCache_.getAvailableLumiPrincipalPtr();
    assert(lbp);
    lbp->setAux(*input_->luminosityBlockAuxiliary());
    {
      SendSourceTerminationSignalIfException sentry(actReg_.get());
      input_->readLuminosityBlock(*lbp, *historyAppender_);
      sentry.completedSuccessfully();
    }
    lbp->setRunPrincipal(std::move(rp));
    return lbp;
  }

  void EventProcessor::readAndMergeLumi(LuminosityBlockProcessingStatus& iStatus) {
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
                                     RunPrincipal const& runPrincipal,
                                     MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    using namespace edm::waiting_task;
    if (runPrincipal.shouldWriteRun() != RunPrincipal::kNo) {
      chain::first([&](auto nextTask) {
        ServiceRegistry::Operate op(serviceToken_);
        schedule_->writeRunAsync(nextTask, runPrincipal, &processContext_, actReg_.get(), mergeableRunProductMetadata);
      }) | chain::ifThen(not subProcesses_.empty(), [this, &runPrincipal, mergeableRunProductMetadata](auto nextTask) {
        ServiceRegistry::Operate op(serviceToken_);
        for (auto& s : subProcesses_) {
          s.writeRunAsync(nextTask, runPrincipal, mergeableRunProductMetadata);
        }
      }) | chain::runLast(std::move(task));
    }
  }

  void EventProcessor::clearRunPrincipal(RunProcessingStatus& iStatus) {
    for (auto& s : subProcesses_) {
      s.clearRunPrincipal(*iStatus.runPrincipal());
    }
    iStatus.runPrincipal()->setShouldWriteRun(RunPrincipal::kUninitialized);
    iStatus.runPrincipal()->clearPrincipal();
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

  void EventProcessor::clearLumiPrincipal(LuminosityBlockProcessingStatus& iStatus) {
    for (auto& s : subProcesses_) {
      s.clearLumiPrincipal(*iStatus.lumiPrincipal());
    }
    iStatus.lumiPrincipal()->setRunPrincipal(std::shared_ptr<RunPrincipal>());
    iStatus.lumiPrincipal()->setShouldWriteLumi(LuminosityBlockPrincipal::kUninitialized);
    iStatus.lumiPrincipal()->clearPrincipal();
  }

  void EventProcessor::readAndMergeRunEntriesAsync(std::shared_ptr<RunProcessingStatus> iRunStatus,
                                                   WaitingTaskHolder iHolder) {
    auto group = iHolder.group();
    sourceResourcesAcquirer_.serialQueueChain().push(
        *group, [this, status = std::move(iRunStatus), holder = std::move(iHolder)]() mutable {
          CMS_SA_ALLOW try {
            ServiceRegistry::Operate operate(serviceToken_);

            std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));

            nextTransitionType();
            while (lastTransitionType() == InputSource::IsRun and status->runPrincipal()->run() == input_->run() and
                   status->runPrincipal()->reducedProcessHistoryID() == input_->reducedProcessHistoryID()) {
              if (status->holderOfTaskInProcessRuns().taskHasFailed()) {
                status->setStopBeforeProcessingRun(true);
                return;
              }
              readAndMergeRun(*status);
              nextTransitionType();
            }
          } catch (...) {
            status->setStopBeforeProcessingRun(true);
            holder.doneWaiting(std::current_exception());
          }
        });
  }

  void EventProcessor::readAndMergeLumiEntriesAsync(std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus,
                                                    WaitingTaskHolder iHolder) {
    auto group = iHolder.group();
    sourceResourcesAcquirer_.serialQueueChain().push(
        *group, [this, iLumiStatus = std::move(iLumiStatus), holder = std::move(iHolder)]() mutable {
          CMS_SA_ALLOW try {
            ServiceRegistry::Operate operate(serviceToken_);

            std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));

            nextTransitionType();
            while (lastTransitionType() == InputSource::IsLumi and
                   iLumiStatus->lumiPrincipal()->luminosityBlock() == input_->luminosityBlock()) {
              readAndMergeLumi(*iLumiStatus);
              nextTransitionType();
            }
          } catch (...) {
            holder.doneWaiting(std::current_exception());
          }
        });
  }

  void EventProcessor::handleNextItemAfterMergingRunEntries(std::shared_ptr<RunProcessingStatus> iRunStatus,
                                                            WaitingTaskHolder iHolder) {
    if (lastTransitionType() == InputSource::IsFile) {
      iRunStatus->holderOfTaskInProcessRuns().doneWaiting(std::exception_ptr{});
      iHolder.doneWaiting(std::exception_ptr{});
    } else if (lastTransitionType() == InputSource::IsLumi && !iHolder.taskHasFailed()) {
      CMS_SA_ALLOW try {
        beginLumiAsync(IOVSyncValue(EventID(input_->run(), input_->luminosityBlock(), 0),
                                    input_->luminosityBlockAuxiliary()->beginTime()),
                       iRunStatus,
                       iHolder);
      } catch (...) {
        WaitingTaskHolder copyHolder(iHolder);
        iHolder.doneWaiting(std::current_exception());
        endRunAsync(std::move(iRunStatus), std::move(iHolder));
      }
    } else {
      // Note that endRunAsync will call beginRunAsync for the following run
      // if appropriate.
      endRunAsync(std::move(iRunStatus), std::move(iHolder));
    }
  }

  bool EventProcessor::readNextEventForStream(WaitingTaskHolder const& iTask,
                                              unsigned int iStreamIndex,
                                              LuminosityBlockProcessingStatus& iStatus) {
    // This function returns true if it successfully reads an event for the stream and that
    // requires both that an event is next and there are no problems or requests to stop.

    if (iTask.taskHasFailed()) {
      // We want all streams to stop or all streams to pause. If we are already in the
      // middle of pausing streams, then finish pausing all of them and the lumi will be
      // ended later. Otherwise, just end it now.
      if (iStatus.eventProcessingState() == LuminosityBlockProcessingStatus::EventProcessingState::kProcessing) {
        iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
      }
      return false;
    }

    // Did another stream already stop or pause this lumi?
    if (iStatus.eventProcessingState() != LuminosityBlockProcessingStatus::EventProcessingState::kProcessing) {
      return false;
    }

    // Are output modules or the looper requesting we stop?
    if (shouldWeStop()) {
      lastSourceTransition_ = InputSource::IsStop;
      iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
      return false;
    }

    ServiceRegistry::Operate operate(serviceToken_);

    // need to use lock in addition to the serial task queue because
    // of delayed provenance reading and reading data in response to
    // edm::Refs etc
    std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));

    // If we didn't already call nextTransitionType while merging lumis, call it here.
    // This asks the input source what is next and also checks for signals.

    InputSource::ItemType itemType = firstItemAfterLumiMerge_ ? lastTransitionType() : nextTransitionType();
    firstItemAfterLumiMerge_ = false;

    if (InputSource::IsEvent != itemType) {
      // IsFile may continue processing the lumi and
      // looper_ can cause the input source to declare a new IsRun which is actually
      // just a continuation of the previous run
      if (InputSource::IsStop == itemType or InputSource::IsLumi == itemType or
          (InputSource::IsRun == itemType and
           (iStatus.lumiPrincipal()->run() != input_->run() or
            iStatus.lumiPrincipal()->runPrincipal().reducedProcessHistoryID() != input_->reducedProcessHistoryID()))) {
        iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
      } else {
        iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kPauseForFileTransition);
      }
      return false;
    }
    readEvent(iStreamIndex);
    return true;
  }

  void EventProcessor::handleNextEventForStreamAsync(WaitingTaskHolder iTask, unsigned int iStreamIndex) {
    auto group = iTask.group();
    sourceResourcesAcquirer_.serialQueueChain().push(*group, [this, iTask = std::move(iTask), iStreamIndex]() mutable {
      CMS_SA_ALLOW try {
        auto status = streamLumiStatus_[iStreamIndex].get();
        ServiceRegistry::Operate operate(serviceToken_);

        if (readNextEventForStream(iTask, iStreamIndex, *status)) {
          auto recursionTask =
              make_waiting_task([this, iTask, iStreamIndex](std::exception_ptr const* iEventException) mutable {
                if (iEventException) {
                  WaitingTaskHolder copyHolder(iTask);
                  copyHolder.doneWaiting(*iEventException);
                  // Intentionally, we don't return here. The recursive call to
                  // handleNextEvent takes care of immediately ending the run properly
                  // using the same code it uses to end the run in other situations.
                }
                handleNextEventForStreamAsync(std::move(iTask), iStreamIndex);
              });

          processEventAsync(WaitingTaskHolder(*iTask.group(), recursionTask), iStreamIndex);
        } else {
          // the stream will stop processing this lumi now
          if (status->eventProcessingState() == LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi) {
            if (not status->haveStartedNextLumiOrEndedRun()) {
              status->startNextLumiOrEndRun();
              if (lastTransitionType() == InputSource::IsLumi && !iTask.taskHasFailed()) {
                CMS_SA_ALLOW try {
                  beginLumiAsync(IOVSyncValue(EventID(input_->run(), input_->luminosityBlock(), 0),
                                              input_->luminosityBlockAuxiliary()->beginTime()),
                                 streamRunStatus_[iStreamIndex],
                                 iTask);
                } catch (...) {
                  WaitingTaskHolder copyHolder(iTask);
                  copyHolder.doneWaiting(std::current_exception());
                  endRunAsync(streamRunStatus_[iStreamIndex], iTask);
                }
              } else {
                // If appropriate, this will also start the next run.
                endRunAsync(streamRunStatus_[iStreamIndex], iTask);
              }
            }
            streamEndLumiAsync(iTask, iStreamIndex);
          } else {
            assert(status->eventProcessingState() ==
                   LuminosityBlockProcessingStatus::EventProcessingState::kPauseForFileTransition);
            auto runStatus = streamRunStatus_[iStreamIndex].get();

            if (runStatus->holderOfTaskInProcessRuns().hasTask()) {
              runStatus->holderOfTaskInProcessRuns().doneWaiting(std::exception_ptr{});
            }
          }
        }
      } catch (...) {
        WaitingTaskHolder copyHolder(iTask);
        copyHolder.doneWaiting(std::current_exception());
        handleNextEventForStreamAsync(std::move(iTask), iStreamIndex);
      }
    });
  }

  void EventProcessor::readEvent(unsigned int iStreamIndex) {
    //TODO this will have to become per stream
    auto& event = principalCache_.eventPrincipal(iStreamIndex);
    StreamContext streamContext(event.streamID(), &processContext_);

    SendSourceTerminationSignalIfException sentry(actReg_.get());
    input_->readEvent(event, streamContext);

    streamRunStatus_[iStreamIndex]->updateLastTimestamp(input_->timestamp());
    streamLumiStatus_[iStreamIndex]->updateLastTimestamp(input_->timestamp());
    sentry.completedSuccessfully();

    FDEBUG(1) << "\treadEvent\n";
  }

  void EventProcessor::processEventAsync(WaitingTaskHolder iHolder, unsigned int iStreamIndex) {
    iHolder.group()->run([this, iHolder, iStreamIndex]() { processEventAsyncImpl(iHolder, iStreamIndex); });
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

  void EventProcessor::setExceptionMessageRuns() { exceptionMessageRuns_ = true; }

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

  void EventProcessor::warnAboutModulesRequiringRunSynchronization() const {
    std::unique_ptr<LogSystem> s;
    for (auto worker : schedule_->allWorkers()) {
      if (worker->wantsGlobalRuns() and worker->globalRunsQueue()) {
        if (not s) {
          s = std::make_unique<LogSystem>("ModulesSynchingOnRuns");
          (*s) << "The following modules require synchronizing on Run boundaries:";
        }
        (*s) << "\n  " << worker->description()->moduleName() << " " << worker->description()->moduleLabel();
      }
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
