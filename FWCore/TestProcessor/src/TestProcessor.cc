// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     TestProcessor
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:08 GMT
//

// system include files

// user include files
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/TestProcessor/interface/EventSetupTestHelper.h"

#include "FWCore/Framework/interface/ScheduleItems.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/globalTransitionAsync.h"
#include "FWCore/Framework/src/streamTransitionAsync.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/DelayedReader.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"

#include "FWCore/Utilities/interface/ExceptionCollector.h"

#include "FWCore/Concurrency/interface/ThreadsController.h"

#include "DataFormats/Provenance/interface/ParentageRegistry.h"

#define xstr(s) str(s)
#define str(s) #s

namespace edm {
  namespace test {

    namespace {

      bool oneTimeInitializationImpl() {
        edmplugin::PluginManager::configure(edmplugin::standard::config());

        static std::unique_ptr<edm::ThreadsController> tsiPtr = std::make_unique<edm::ThreadsController>(1);

        // register the empty parentage vector , once and for all
        ParentageRegistry::instance()->insertMapped(Parentage());

        // register the empty parameter set, once and for all.
        ParameterSet().registerIt();
        return true;
      }

      bool oneTimeInitialization() {
        static const bool s_init{oneTimeInitializationImpl()};
        return s_init;
      }
    }  // namespace

    //
    // constructors and destructor
    //
    TestProcessor::TestProcessor(Config const& iConfig, ServiceToken iToken)
        : globalControl_(tbb::global_control::max_allowed_parallelism, 1),
          arena_(1),
          espController_(std::make_unique<eventsetup::EventSetupsController>()),
          historyAppender_(std::make_unique<HistoryAppender>()),
          moduleRegistry_(std::make_shared<ModuleRegistry>()) {
      //Setup various singletons
      (void)oneTimeInitialization();

      ProcessDescImpl desc(iConfig.pythonConfiguration());

      auto psetPtr = desc.parameterSet();

      validateTopLevelParameterSets(psetPtr.get());

      labelOfTestModule_ = psetPtr->getParameter<std::string>("@moduleToTest");

      auto procDesc = desc.processDesc();
      // Now do general initialization
      ScheduleItems items;

      //initialize the services
      auto& serviceSets = procDesc->getServicesPSets();
      ServiceToken token = items.initServices(serviceSets, *psetPtr, iToken, serviceregistry::kOverlapIsError, true);
      serviceToken_ = items.addCPRandTNS(*psetPtr, token);

      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      // intialize miscellaneous items
      std::shared_ptr<CommonParams> common(items.initMisc(*psetPtr));

      // intialize the event setup provider
      esp_ = espController_->makeProvider(*psetPtr, items.actReg_.get());

      auto nThreads = 1U;
      auto nStreams = 1U;
      auto nConcurrentLumis = 1U;
      auto nConcurrentRuns = 1U;
      preallocations_ = PreallocationConfiguration{nThreads, nStreams, nConcurrentLumis, nConcurrentRuns};

      espController_->setMaxConcurrentIOVs(nStreams, nConcurrentLumis);
      if (not iConfig.esProduceEntries().empty()) {
        esHelper_ = std::make_unique<EventSetupTestHelper>(iConfig.esProduceEntries());
        esp_->add(std::dynamic_pointer_cast<eventsetup::DataProxyProvider>(esHelper_));
        esp_->add(std::dynamic_pointer_cast<EventSetupRecordIntervalFinder>(esHelper_));
      }

      preg_ = items.preg();
      processConfiguration_ = items.processConfiguration();

      edm::ParameterSet emptyPSet;
      emptyPSet.registerIt();
      auto psetid = emptyPSet.id();

      ProcessHistory oldHistory;
      for (auto const& p : iConfig.extraProcesses()) {
        oldHistory.emplace_back(p, psetid, xstr(PROJECT_VERSION), "0");
        processHistoryRegistry_.registerProcessHistory(oldHistory);
      }

      //setup the products we will be adding to the event
      for (auto const& produce : iConfig.produceEntries()) {
        auto processName = produce.processName_;
        if (processName.empty()) {
          processName = processConfiguration_->processName();
        }
        edm::TypeWithDict twd(produce.type_.typeInfo());
        edm::BranchDescription product(edm::InEvent,
                                       produce.moduleLabel_,
                                       processName,
                                       twd.userClassName(),
                                       twd.friendlyClassName(),
                                       produce.instanceLabel_,
                                       "",
                                       psetid,
                                       twd,
                                       true  //force this to come from 'source'
        );
        product.init();
        dataProducts_.emplace_back(product, std::unique_ptr<WrapperBase>());
        preg_->addProduct(product);
      }

      schedule_ = items.initSchedule(*psetPtr, false, preallocations_, &processContext_);
      // set the data members
      act_table_ = std::move(items.act_table_);
      actReg_ = items.actReg_;
      branchIDListHelper_ = items.branchIDListHelper();
      thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
      processContext_.setProcessConfiguration(processConfiguration_.get());
      principalCache_.setProcessHistoryRegistry(processHistoryRegistry_);

      principalCache_.setNumberOfConcurrentPrincipals(preallocations_);

      preg_->setFrozen();
      for (unsigned int index = 0; index < preallocations_.numberOfStreams(); ++index) {
        // Reusable event principal
        auto ep = std::make_shared<EventPrincipal>(preg_,
                                                   branchIDListHelper_,
                                                   thinnedAssociationsHelper_,
                                                   *processConfiguration_,
                                                   historyAppender_.get(),
                                                   index);
        principalCache_.insert(std::move(ep));
      }

      for (unsigned int index = 0; index < preallocations_.numberOfLuminosityBlocks(); ++index) {
        auto lp =
            std::make_unique<LuminosityBlockPrincipal>(preg_, *processConfiguration_, historyAppender_.get(), index);
        principalCache_.insert(std::move(lp));
      }
      {
        auto pb = std::make_unique<ProcessBlockPrincipal>(preg_, *processConfiguration_);
        principalCache_.insert(std::move(pb));
      }
    }

    TestProcessor::~TestProcessor() noexcept(false) { teardownProcessing(); }
    //
    // member functions
    //

    void TestProcessor::put(unsigned int index, std::unique_ptr<WrapperBase> iWrapper) {
      if (index >= dataProducts_.size()) {
        throw cms::Exception("LogicError") << "Products must be declared to the TestProcessor::Config object\n"
                                              "with a call to the function \'produces\' BEFORE passing the\n"
                                              "TestProcessor::Config object to the TestProcessor constructor";
      }
      dataProducts_[index].second = std::move(iWrapper);
    }

    edm::test::Event TestProcessor::testImpl() {
      bool result = arena_.execute([this]() {
        setupProcessing();
        event();

        return schedule_->totalEventsPassed() > 0;
      });
      schedule_->clearCounters();
      if (esHelper_) {
        //We want each test to have its own ES data products
        esHelper_->resetAllProxies();
      }
      return edm::test::Event(
          principalCache_.eventPrincipal(0), labelOfTestModule_, processConfiguration_->processName(), result);
    }

    edm::test::LuminosityBlock TestProcessor::testBeginLuminosityBlockImpl(edm::LuminosityBlockNumber_t iNum) {
      arena_.execute([this, iNum]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        if (not beginProcessBlockCalled_) {
          beginProcessBlock();
        }
        if (not beginRunCalled_) {
          beginRun();
        }
        if (beginLumiCalled_) {
          endLuminosityBlock();
          assert(lumiNumber_ != iNum);
        }
        lumiNumber_ = iNum;
        beginLuminosityBlock();
      });

      if (esHelper_) {
        //We want each test to have its own ES data products
        esHelper_->resetAllProxies();
      }

      return edm::test::LuminosityBlock(lumiPrincipal_, labelOfTestModule_, processConfiguration_->processName());
    }

    edm::test::LuminosityBlock TestProcessor::testEndLuminosityBlockImpl() {
      //using a return value from arena_.execute lead to double delete of shared_ptr
      // based on valgrind output when exception occurred. Use lambda capture instead.
      std::shared_ptr<edm::LuminosityBlockPrincipal> lumi;
      arena_.execute([this, &lumi]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        if (not beginProcessBlockCalled_) {
          beginProcessBlock();
        }
        if (not beginRunCalled_) {
          beginRun();
        }
        if (not beginLumiCalled_) {
          beginLuminosityBlock();
        }
        lumi = endLuminosityBlock();
      });
      if (esHelper_) {
        //We want each test to have its own ES data products
        esHelper_->resetAllProxies();
      }

      return edm::test::LuminosityBlock(std::move(lumi), labelOfTestModule_, processConfiguration_->processName());
    }

    edm::test::Run TestProcessor::testBeginRunImpl(edm::RunNumber_t iNum) {
      arena_.execute([this, iNum]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        if (not beginProcessBlockCalled_) {
          beginProcessBlock();
        }
        if (beginRunCalled_) {
          assert(runNumber_ != iNum);
          endRun();
        }
        runNumber_ = iNum;
        beginRun();
      });
      if (esHelper_) {
        //We want each test to have its own ES data products
        esHelper_->resetAllProxies();
      }

      return edm::test::Run(
          principalCache_.runPrincipalPtr(), labelOfTestModule_, processConfiguration_->processName());
    }
    edm::test::Run TestProcessor::testEndRunImpl() {
      //using a return value from arena_.execute lead to double delete of shared_ptr
      // based on valgrind output when exception occurred. Use lambda capture instead.
      std::shared_ptr<edm::RunPrincipal> rp;
      arena_.execute([this, &rp]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        if (not beginProcessBlockCalled_) {
          beginProcessBlock();
        }
        if (not beginRunCalled_) {
          beginRun();
        }
        rp = endRun();
      });
      if (esHelper_) {
        //We want each test to have its own ES data products
        esHelper_->resetAllProxies();
      }

      return edm::test::Run(rp, labelOfTestModule_, processConfiguration_->processName());
    }

    edm::test::ProcessBlock TestProcessor::testBeginProcessBlockImpl() {
      arena_.execute([this]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        beginProcessBlock();
      });
      return edm::test::ProcessBlock(
          &principalCache_.processBlockPrincipal(), labelOfTestModule_, processConfiguration_->processName());
    }
    edm::test::ProcessBlock TestProcessor::testEndProcessBlockImpl() {
      auto pbp = arena_.execute([this]() {
        if (not beginJobCalled_) {
          beginJob();
        }
        if (not beginProcessBlockCalled_) {
          beginProcessBlock();
        }
        return endProcessBlock();
      });
      return edm::test::ProcessBlock(pbp, labelOfTestModule_, processConfiguration_->processName());
    }

    void TestProcessor::setupProcessing() {
      if (not beginJobCalled_) {
        beginJob();
      }
      if (not beginProcessBlockCalled_) {
        beginProcessBlock();
      }
      if (not beginRunCalled_) {
        beginRun();
      }
      if (not beginLumiCalled_) {
        beginLuminosityBlock();
      }
    }

    void TestProcessor::teardownProcessing() {
      arena_.execute([this]() {
        if (beginLumiCalled_) {
          endLuminosityBlock();
          beginLumiCalled_ = false;
        }
        if (beginRunCalled_) {
          endRun();
          beginRunCalled_ = false;
        }
        if (beginProcessBlockCalled_) {
          endProcessBlock();
          beginProcessBlockCalled_ = false;
        }
        if (beginJobCalled_) {
          endJob();
        }
        espController_->endIOVs();
      });
    }

    void TestProcessor::beginJob() {
      ServiceRegistry::Operate operate(serviceToken_);

      service::SystemBounds bounds(preallocations_.numberOfStreams(),
                                   preallocations_.numberOfLuminosityBlocks(),
                                   preallocations_.numberOfRuns(),
                                   preallocations_.numberOfThreads());
      actReg_->preallocateSignal_(bounds);
      schedule_->convertCurrentProcessAlias(processConfiguration_->processName());
      PathsAndConsumesOfModules pathsAndConsumesOfModules;

      //The code assumes only modules make data in the current process
      // Since the test os also allowed to do so, it can lead to problems.
      //pathsAndConsumesOfModules.initialize(schedule_.get(), preg_);

      //NOTE: this may throw
      //checkForModuleDependencyCorrectness(pathsAndConsumesOfModules, false);
      actReg_->preBeginJobSignal_(pathsAndConsumesOfModules, processContext_);

      espController_->finishConfiguration();

      schedule_->beginJob(*preg_, esp_->recordsToProxyIndices());
      actReg_->postBeginJobSignal_();

      for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
        schedule_->beginStream(i);
      }
      beginJobCalled_ = true;
    }

    void TestProcessor::beginProcessBlock() {
      ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();
      processBlockPrincipal.fillProcessBlockPrincipal(processConfiguration_->processName());

      std::vector<edm::SubProcess> emptyList;
      {
        ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
        using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>;
        auto globalWaitTask = make_empty_waiting_task();
        globalWaitTask->increment_ref_count();

        beginGlobalTransitionAsync<Traits>(
            WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList);

        globalWaitTask->wait_for_all();
        if (globalWaitTask->exceptionPtr() != nullptr) {
          std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
        }
      }
      beginProcessBlockCalled_ = true;
    }

    void TestProcessor::beginRun() {
      ProcessHistoryID phid;
      auto aux = std::make_shared<RunAuxiliary>(runNumber_, Timestamp(), Timestamp());
      auto rp = std::make_shared<RunPrincipal>(aux, preg_, *processConfiguration_, historyAppender_.get(), 0);

      principalCache_.insert(rp);
      RunPrincipal& runPrincipal = principalCache_.runPrincipal(phid, runNumber_);

      IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0), runPrincipal.beginTime());
      espController_->eventSetupForInstance(ts);

      auto const& es = esp_->eventSetupImpl();

      RunTransitionInfo transitionInfo(runPrincipal, es);

      std::vector<edm::SubProcess> emptyList;
      {
        using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>;
        auto globalWaitTask = make_empty_waiting_task();
        globalWaitTask->increment_ref_count();
        beginGlobalTransitionAsync<Traits>(
            WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList);
        globalWaitTask->wait_for_all();
        if (globalWaitTask->exceptionPtr() != nullptr) {
          std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
        }
      }
      {
        //To wait, the ref count has to be 1+#streams
        auto streamLoopWaitTask = make_empty_waiting_task();
        streamLoopWaitTask->increment_ref_count();

        using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>;
        beginStreamsTransitionAsync<Traits>(WaitingTaskHolder(streamLoopWaitTask.get()),
                                            *schedule_,
                                            preallocations_.numberOfStreams(),
                                            transitionInfo,
                                            serviceToken_,
                                            emptyList);

        streamLoopWaitTask->wait_for_all();
        if (streamLoopWaitTask->exceptionPtr() != nullptr) {
          std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
        }
      }
      beginRunCalled_ = true;
    }

    void TestProcessor::beginLuminosityBlock() {
      LuminosityBlockAuxiliary aux(runNumber_, lumiNumber_, Timestamp(), Timestamp());
      lumiPrincipal_ = principalCache_.getAvailableLumiPrincipalPtr();
      lumiPrincipal_->clearPrincipal();
      assert(lumiPrincipal_);
      lumiPrincipal_->setAux(aux);

      lumiPrincipal_->setRunPrincipal(principalCache_.runPrincipalPtr());

      IOVSyncValue ts(EventID(runNumber_, lumiNumber_, eventNumber_), lumiPrincipal_->beginTime());
      espController_->eventSetupForInstance(ts);

      auto const& es = esp_->eventSetupImpl();

      LumiTransitionInfo transitionInfo(*lumiPrincipal_, es, nullptr);

      std::vector<edm::SubProcess> emptyList;
      {
        using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>;
        auto globalWaitTask = make_empty_waiting_task();
        globalWaitTask->increment_ref_count();
        beginGlobalTransitionAsync<Traits>(
            WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList);
        globalWaitTask->wait_for_all();
        if (globalWaitTask->exceptionPtr() != nullptr) {
          std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
        }
      }
      {
        //To wait, the ref count has to be 1+#streams
        auto streamLoopWaitTask = make_empty_waiting_task();
        streamLoopWaitTask->increment_ref_count();

        using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>;

        beginStreamsTransitionAsync<Traits>(WaitingTaskHolder(streamLoopWaitTask.get()),
                                            *schedule_,
                                            preallocations_.numberOfStreams(),
                                            transitionInfo,
                                            serviceToken_,
                                            emptyList);

        streamLoopWaitTask->wait_for_all();
        if (streamLoopWaitTask->exceptionPtr() != nullptr) {
          std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
        }
      }
      beginLumiCalled_ = true;
    }

    void TestProcessor::event() {
      auto pep = &(principalCache_.eventPrincipal(0));

      //this resets the EventPrincipal (if it had been used before)
      pep->clearEventPrincipal();
      pep->fillEventPrincipal(
          edm::EventAuxiliary(EventID(runNumber_, lumiNumber_, eventNumber_), "", Timestamp(), false),
          nullptr,
          nullptr);
      assert(lumiPrincipal_.get() != nullptr);
      pep->setLuminosityBlockPrincipal(lumiPrincipal_.get());

      for (auto& p : dataProducts_) {
        if (p.second) {
          pep->put(p.first, std::move(p.second), ProductProvenance(p.first.branchID()));
        } else {
          //The data product was not set so we need to
          // tell the ProductResolver not to wait
          auto r = pep->getProductResolver(p.first.branchID());
          r->putProduct(std::unique_ptr<WrapperBase>());
        }
      }

      ServiceRegistry::Operate operate(serviceToken_);

      auto waitTask = make_empty_waiting_task();
      waitTask->increment_ref_count();

      EventTransitionInfo info(*pep, esp_->eventSetupImpl());
      schedule_->processOneEventAsync(edm::WaitingTaskHolder(waitTask.get()), 0, info, serviceToken_);

      waitTask->wait_for_all();
      if (waitTask->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(waitTask->exceptionPtr()));
      }
      ++eventNumber_;
    }

    std::shared_ptr<LuminosityBlockPrincipal> TestProcessor::endLuminosityBlock() {
      auto lumiPrincipal = lumiPrincipal_;
      if (beginLumiCalled_) {
        beginLumiCalled_ = false;
        lumiPrincipal_.reset();

        IOVSyncValue ts(EventID(runNumber_, lumiNumber_, eventNumber_), lumiPrincipal->endTime());
        espController_->eventSetupForInstance(ts);

        auto const& es = esp_->eventSetupImpl();

        LumiTransitionInfo transitionInfo(*lumiPrincipal, es, nullptr);

        std::vector<edm::SubProcess> emptyList;

        //To wait, the ref count has to be 1+#streams
        {
          auto streamLoopWaitTask = make_empty_waiting_task();
          streamLoopWaitTask->increment_ref_count();

          using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;

          endStreamsTransitionAsync<Traits>(WaitingTaskHolder(streamLoopWaitTask.get()),
                                            *schedule_,
                                            preallocations_.numberOfStreams(),
                                            transitionInfo,
                                            serviceToken_,
                                            emptyList,
                                            false);

          streamLoopWaitTask->wait_for_all();
          if (streamLoopWaitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
          }
        }
        {
          auto globalWaitTask = make_empty_waiting_task();
          globalWaitTask->increment_ref_count();

          using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>;
          endGlobalTransitionAsync<Traits>(
              WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList, false);
          globalWaitTask->wait_for_all();
          if (globalWaitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
          }
        }
      }
      return lumiPrincipal;
    }

    std::shared_ptr<edm::RunPrincipal> TestProcessor::endRun() {
      std::shared_ptr<RunPrincipal> rp;
      if (beginRunCalled_) {
        beginRunCalled_ = false;
        ProcessHistoryID phid;
        rp = principalCache_.runPrincipalPtr(phid, runNumber_);
        RunPrincipal& runPrincipal = *rp;

        IOVSyncValue ts(
            EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
            runPrincipal.endTime());
        espController_->eventSetupForInstance(ts);

        auto const& es = esp_->eventSetupImpl();

        RunTransitionInfo transitionInfo(runPrincipal, es);

        std::vector<edm::SubProcess> emptyList;

        //To wait, the ref count has to be 1+#streams
        {
          auto streamLoopWaitTask = make_empty_waiting_task();
          streamLoopWaitTask->increment_ref_count();

          using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>;

          endStreamsTransitionAsync<Traits>(WaitingTaskHolder(streamLoopWaitTask.get()),
                                            *schedule_,
                                            preallocations_.numberOfStreams(),
                                            transitionInfo,
                                            serviceToken_,
                                            emptyList,
                                            false);

          streamLoopWaitTask->wait_for_all();
          if (streamLoopWaitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(streamLoopWaitTask->exceptionPtr()));
          }
        }
        {
          auto globalWaitTask = make_empty_waiting_task();
          globalWaitTask->increment_ref_count();

          using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>;
          endGlobalTransitionAsync<Traits>(
              WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList, false);
          globalWaitTask->wait_for_all();
          if (globalWaitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
          }
        }

        principalCache_.deleteRun(phid, runNumber_);
      }
      return rp;
    }

    ProcessBlockPrincipal const* TestProcessor::endProcessBlock() {
      ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();
      if (beginProcessBlockCalled_) {
        beginProcessBlockCalled_ = false;

        std::vector<edm::SubProcess> emptyList;
        {
          auto globalWaitTask = make_empty_waiting_task();
          globalWaitTask->increment_ref_count();

          ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
          using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
          endGlobalTransitionAsync<Traits>(
              WaitingTaskHolder(globalWaitTask.get()), *schedule_, transitionInfo, serviceToken_, emptyList, false);
          globalWaitTask->wait_for_all();
          if (globalWaitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
          }
        }
      }
      return &processBlockPrincipal;
    }

    void TestProcessor::endJob() {
      if (!beginJobCalled_) {
        return;
      }
      beginJobCalled_ = false;

      // Collects exceptions, so we don't throw before all operations are performed.
      ExceptionCollector c(
          "Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      //NOTE: this really should go elsewhere in the future
      for (unsigned int i = 0; i < preallocations_.numberOfStreams(); ++i) {
        c.call([this, i]() { this->schedule_->endStream(i); });
      }
      auto actReg = actReg_.get();
      c.call([actReg]() { actReg->preEndJobSignal_(); });
      schedule_->endJob(c);
      c.call([actReg]() { actReg->postEndJobSignal_(); });
      if (c.hasThrown()) {
        c.rethrow();
      }
    }

    void TestProcessor::setRunNumber(edm::RunNumber_t iRun) {
      if (beginRunCalled_) {
        endLuminosityBlock();
        endRun();
      }
      runNumber_ = iRun;
    }
    void TestProcessor::setLuminosityBlockNumber(edm::LuminosityBlockNumber_t iLumi) {
      endLuminosityBlock();
      lumiNumber_ = iLumi;
    }

    void TestProcessor::setEventNumber(edm::EventNumber_t iEv) { eventNumber_ = iEv; }

  }  // namespace test
}  // namespace edm
