/*
 *  proxyfactoryproducer_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 28-Jun-05
 */
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "oneapi/tbb/global_control.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/OutputModuleCommunicatorT.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProductResolversFactory.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "makeEmptyEventSetupImplForTest.h"

#include "catch2/catch_all.hpp"

namespace edm {
  class ModuleCallingContext;
}

namespace {
  enum class Trans {
    kBeginJob,
    kGlobalOpenInputFile,
    kGlobalBeginRun,
    kGlobalBeginRunProduce,
    kGlobalBeginLuminosityBlock,
    kEvent,
    kGlobalEndLuminosityBlock,
    kGlobalEndRun,
    kGlobalCloseInputFile,
    kEndJob
  };

  typedef std::vector<Trans> Expectations;

  edm::ActivityRegistry activityRegistry;

  struct ShadowStreamID {
    constexpr ShadowStreamID() : value(0) {}
    unsigned int value;
  };

  union IDUnion {
    IDUnion() : m_shadow() {}
    ShadowStreamID m_shadow;
    edm::StreamID m_id;
  };
  static edm::StreamID makeID() {
    IDUnion u;
    assert(u.m_id.value() == 0);
    return u.m_id;
  }

  static const edm::StreamID s_streamID0 = makeID();

  struct DummyCache {};

  class BasicOutputModule : public edm::one::OutputModule<> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    BasicOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<>(iPSet) {}
    unsigned int m_count = 0;

    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }
  };

  class RunOutputModule : public edm::one::OutputModule<edm::one::WatchRuns> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    RunOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::one::WatchRuns>(iPSet) {}
    unsigned int m_count = 0;
    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    void beginRun(edm::RunForOutput const&) override { ++m_count; }

    void endRun(edm::RunForOutput const&) override { ++m_count; }
  };

  class LumiOutputModule : public edm::one::OutputModule<edm::one::WatchLuminosityBlocks> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    LumiOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::one::WatchLuminosityBlocks>(iPSet) {}
    unsigned int m_count = 0;
    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }
  };

  class RunCacheOutputModule : public edm::one::OutputModule<edm::RunCache<DummyCache>> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    RunCacheOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::RunCache<DummyCache>>(iPSet) {}
    mutable std::atomic<unsigned int> m_count = 0;
    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    std::shared_ptr<DummyCache> globalBeginRun(edm::RunForOutput const&) const override {
      ++m_count;
      return std::shared_ptr<DummyCache>{};
    }

    void globalEndRun(edm::RunForOutput const&) override { ++m_count; }
  };

  class LumiCacheOutputModule : public edm::one::OutputModule<edm::LuminosityBlockCache<DummyCache>> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    LumiCacheOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::LuminosityBlockCache<DummyCache>>(iPSet) {}
    mutable std::atomic<unsigned int> m_count = 0;
    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    std::shared_ptr<DummyCache> globalBeginLuminosityBlock(edm::LuminosityBlockForOutput const&) const override {
      ++m_count;
      return std::shared_ptr<DummyCache>{};
    }

    void globalEndLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }
  };

  class FileOutputModule : public edm::one::OutputModule<edm::WatchInputFiles> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    FileOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::WatchInputFiles>(iPSet) {}
    unsigned int m_count = 0;
    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }

    void respondToOpenInputFile(edm::FileBlock const&) override { ++m_count; }

    void respondToCloseInputFile(edm::FileBlock const&) override { ++m_count; }
  };

  class ResourceOutputModule : public edm::one::OutputModule<edm::one::SharedResources> {
  public:
    using edm::one::OutputModuleBase::doPreallocate;
    ResourceOutputModule(edm::ParameterSet const& iPSet)
        : edm::one::OutputModuleBase(iPSet), edm::one::OutputModule<edm::one::SharedResources>(iPSet) {
      usesResource("foo");
    }
    unsigned int m_count = 0;

    void write(edm::EventForOutput const&) override { ++m_count; }
    void writeRun(edm::RunForOutput const&) override { ++m_count; }
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override { ++m_count; }
  };

  // Test fixture struct
  struct TestFixture {
    std::map<Trans, std::function<void(edm::Worker*, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator*)>>
        m_transToFunc;

    edm::ProcessConfiguration m_procConfig;
    edm::PreallocationConfiguration m_preallocConfig;
    std::shared_ptr<edm::ProductRegistry> m_prodReg;
    std::shared_ptr<edm::BranchIDListHelper> m_idHelper;
    std::shared_ptr<edm::ThinnedAssociationsHelper> m_associationsHelper;
    std::unique_ptr<edm::EventPrincipal> m_ep;
    edm::HistoryAppender historyAppender_;
    std::shared_ptr<edm::LuminosityBlockPrincipal> m_lbp;
    std::shared_ptr<edm::RunPrincipal> m_rp;
    std::shared_ptr<edm::ActivityRegistry>
        m_actReg;  // We do not use propagate_const because the registry itself is mutable.
    edm::EventSetupImpl m_es = edm::makeEmptyEventSetupImplForTest();
    edm::ModuleDescription m_desc = {"Dummy", "dummy"};
    edm::WorkerParams m_params;

    typedef edm::service::TriggerNamesService TNS;
    typedef edm::serviceregistry::ServiceWrapper<TNS> w_TNS;
    std::shared_ptr<w_TNS> tnsptr_;
    edm::ServiceToken serviceToken_;

    template <typename T>
    void testTransitions(std::shared_ptr<T> iMod, Expectations const& iExpect);

    template <typename Traits, typename Info>
    void doWork(edm::Worker* iBase, Info const& info, edm::StreamID id, edm::ParentContext const& iContext) {
      oneapi::tbb::task_group group;
      edm::FinalWaitingTask task{group};
      edm::ServiceToken token;
      iBase->doWorkAsync<Traits>(edm::WaitingTaskHolder(group, &task), info, token, id, iContext, nullptr);
      task.wait();
    }

    TestFixture()
        : m_prodReg(new edm::ProductRegistry{}),
          m_idHelper(new edm::BranchIDListHelper{}),
          m_associationsHelper(new edm::ThinnedAssociationsHelper{}),
          m_ep() {
      //Setup the principals
      m_prodReg->setFrozen();
      m_idHelper->updateFromRegistry(*m_prodReg);
      edm::EventID eventID = edm::EventID::firstValidEvent();

      std::string uuid = edm::createGlobalIdentifier();
      edm::Timestamp now(1234567UL);
      m_rp.reset(new edm::RunPrincipal(
          m_prodReg, edm::productResolversFactory::makePrimary, m_procConfig, &historyAppender_, 0));
      m_rp->setAux(edm::RunAuxiliary(eventID.run(), now, now));
      edm::LuminosityBlockAuxiliary lumiAux(m_rp->run(), 1, now, now);
      m_lbp.reset(new edm::LuminosityBlockPrincipal(
          m_prodReg, edm::productResolversFactory::makePrimary, m_procConfig, &historyAppender_, 0));
      m_lbp->setAux(lumiAux);
      m_lbp->setRunPrincipal(m_rp);
      edm::EventAuxiliary eventAux(eventID, uuid, now, true);

      m_ep.reset(new edm::EventPrincipal(m_prodReg,
                                         edm::productResolversFactory::makePrimary,
                                         m_idHelper,
                                         m_associationsHelper,
                                         m_procConfig,
                                         nullptr));
      m_ep->fillEventPrincipal(eventAux, nullptr);
      m_ep->setLuminosityBlockPrincipal(m_lbp.get());
      m_actReg.reset(new edm::ActivityRegistry);

      //For each transition, bind a lambda which will call the proper method of the Worker
      m_transToFunc[Trans::kGlobalOpenInputFile] =
          [](edm::Worker* iBase, edm::maker::ModuleHolder* iHolder, edm::OutputModuleCommunicator*) {
            edm::FileBlock fb;
            iHolder->respondToOpenInputFile(fb);
          };

      m_transToFunc[Trans::kGlobalBeginRun] =
          [this](edm::Worker* iBase, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator*) {
            typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalBegin> Traits;
            edm::GlobalContext gc(edm::GlobalContext::Transition::kBeginRun, nullptr);
            edm::ParentContext parentContext(&gc);
            iBase->setActivityRegistry(m_actReg);
            edm::RunTransitionInfo info(*m_rp, m_es);
            doWork<Traits>(iBase, info, edm::StreamID::invalidStreamID(), parentContext);
          };

      m_transToFunc[Trans::kGlobalBeginLuminosityBlock] =
          [this](edm::Worker* iBase, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator*) {
            typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
            edm::GlobalContext gc(edm::GlobalContext::Transition::kBeginLuminosityBlock, nullptr);
            edm::ParentContext parentContext(&gc);
            iBase->setActivityRegistry(m_actReg);
            edm::LumiTransitionInfo info(*m_lbp, m_es);
            doWork<Traits>(iBase, info, edm::StreamID::invalidStreamID(), parentContext);
          };

      m_transToFunc[Trans::kEvent] =
          [this](edm::Worker* iBase, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator*) {
            typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
            edm::StreamContext streamContext(s_streamID0, nullptr);
            edm::ParentContext parentContext(&streamContext);
            iBase->setActivityRegistry(m_actReg);
            edm::EventTransitionInfo info(*m_ep, m_es);
            doWork<Traits>(iBase, info, s_streamID0, parentContext);
          };

      m_transToFunc[Trans::kGlobalEndLuminosityBlock] =
          [this](edm::Worker* iBase, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator* iComm) {
            typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
            edm::GlobalContext gc(edm::GlobalContext::Transition::kEndLuminosityBlock, nullptr);
            edm::ParentContext parentContext(&gc);
            iBase->setActivityRegistry(m_actReg);
            edm::LumiTransitionInfo info(*m_lbp, m_es);
            doWork<Traits>(iBase, info, edm::StreamID::invalidStreamID(), parentContext);
            oneapi::tbb::task_group group;
            edm::FinalWaitingTask task{group};
            iComm->writeLumiAsync(edm::WaitingTaskHolder(group, &task), *m_lbp, nullptr, &activityRegistry);
            task.wait();
          };

      m_transToFunc[Trans::kGlobalEndRun] =
          [this](edm::Worker* iBase, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator* iComm) {
            typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
            edm::GlobalContext gc(edm::GlobalContext::Transition::kEndRun, nullptr);
            edm::ParentContext parentContext(&gc);
            iBase->setActivityRegistry(m_actReg);
            edm::RunTransitionInfo info(*m_rp, m_es);
            doWork<Traits>(iBase, info, edm::StreamID::invalidStreamID(), parentContext);
            oneapi::tbb::task_group group;
            edm::FinalWaitingTask task{group};
            iComm->writeRunAsync(edm::WaitingTaskHolder(group, &task), *m_rp, nullptr, &activityRegistry, nullptr);
            task.wait();
          };

      m_transToFunc[Trans::kGlobalCloseInputFile] =
          [](edm::Worker* iBase, edm::maker::ModuleHolder* iHolder, edm::OutputModuleCommunicator*) {
            edm::FileBlock fb;
            iHolder->respondToCloseInputFile(fb);
          };

      // We want to create the TriggerNamesService because it is used in
      // the tests.  We do that here, but first we need to build a minimal
      // parameter set to pass to its constructor.  Then we build the
      // service and setup the service system.
      edm::ParameterSet proc_pset;

      std::string processName("HLT");
      proc_pset.addParameter<std::string>("@process_name", processName);

      std::vector<std::string> paths;
      edm::ParameterSet trigPaths;
      trigPaths.addParameter<std::vector<std::string>>("@trigger_paths", paths);
      proc_pset.addParameter<edm::ParameterSet>("@trigger_paths", trigPaths);

      std::vector<std::string> endPaths;
      proc_pset.addParameter<std::vector<std::string>>("@end_paths", endPaths);

      // Now create and setup the service
      tnsptr_.reset(new w_TNS(std::make_unique<TNS>(proc_pset)));

      serviceToken_ = edm::ServiceRegistry::createContaining(tnsptr_);
    }
  };

  template <typename T>
  void testTransition(
      std::shared_ptr<T> iMod,
      edm::Worker* iWorker,
      edm::maker::ModuleHolder* h,
      edm::OutputModuleCommunicator* iComm,
      Trans iTrans,
      Expectations const& iExpect,
      std::function<void(edm::Worker*, edm::maker::ModuleHolder*, edm::OutputModuleCommunicator*)> iFunc) {
    assert(0 == iMod->m_count);
    iFunc(iWorker, h, iComm);
    auto count = std::count(iExpect.begin(), iExpect.end(), iTrans);
    if (count != iMod->m_count) {
      std::cout << "For trans " << static_cast<std::underlying_type<Trans>::type>(iTrans) << " expected " << count
                << " and got " << iMod->m_count << std::endl;
    }
    REQUIRE(iMod->m_count == count);
    iMod->m_count = 0;
    iWorker->reset();
  }

  template <typename T>
  void TestFixture::testTransitions(std::shared_ptr<T> iMod, Expectations const& iExpect) {
    oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, 1);

    iMod->doPreallocate(m_preallocConfig);
    edm::WorkerT<edm::one::OutputModuleBase> w{iMod, m_desc, m_params.actions_};
    edm::maker::ModuleHolderT<edm::one::OutputModuleBase> h(iMod);
    h.beginJob();
    edm::OutputModuleCommunicatorT<edm::one::OutputModuleBase> comm(iMod.get());
    for (auto& keyVal : m_transToFunc) {
      testTransition(iMod, &w, &h, &comm, keyVal.first, iExpect, keyVal.second);
    }
  }

  TEST_CASE("One output module tests", "[OneOutputModule]") {
    TestFixture fixture;

    SECTION("basicTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<BasicOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd, {Trans::kEvent, Trans::kGlobalEndLuminosityBlock, Trans::kGlobalEndRun});
    }

    SECTION("runTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<RunOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd,
                              {Trans::kGlobalBeginRun,
                               Trans::kEvent,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndRun,
                               Trans::kGlobalEndRun});
    }

    SECTION("lumiTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<LumiOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd,
                              {Trans::kGlobalBeginLuminosityBlock,
                               Trans::kEvent,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndRun});
    }

    SECTION("runCacheTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<RunCacheOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd,
                              {Trans::kGlobalBeginRun,
                               Trans::kEvent,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndRun,
                               Trans::kGlobalEndRun});
    }

    SECTION("lumiCacheTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<LumiCacheOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd,
                              {Trans::kGlobalBeginLuminosityBlock,
                               Trans::kEvent,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndRun});
    }

    SECTION("fileTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<FileOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd,
                              {Trans::kGlobalOpenInputFile,
                               Trans::kEvent,
                               Trans::kGlobalEndLuminosityBlock,
                               Trans::kGlobalEndRun,
                               Trans::kGlobalCloseInputFile});
    }

    SECTION("resourceTest") {
      //make the services available
      edm::ServiceRegistry::Operate operate(fixture.serviceToken_);

      edm::ParameterSet pset;
      auto testProd = std::make_shared<ResourceOutputModule>(pset);

      REQUIRE(0 == testProd->m_count);
      fixture.testTransitions(testProd, {Trans::kEvent, Trans::kGlobalEndLuminosityBlock, Trans::kGlobalEndRun});
    }
  }
}  // namespace
