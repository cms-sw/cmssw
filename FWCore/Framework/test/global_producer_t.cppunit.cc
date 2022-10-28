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
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

namespace {
  struct ShadowStreamID {
    constexpr ShadowStreamID() : value(0) {}
    unsigned int value;
  };

  union IDUnion {
    IDUnion() : m_shadow() {}
    ShadowStreamID m_shadow;
    edm::StreamID m_id;
  };
}  // namespace
static edm::StreamID makeID() {
  IDUnion u;
  assert(u.m_id.value() == 0);
  return u.m_id;
}
static const edm::StreamID s_streamID0 = makeID();

class testGlobalProducer : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testGlobalProducer);

  CPPUNIT_TEST(basicTest);
  CPPUNIT_TEST(streamTest);
  CPPUNIT_TEST(runTest);
  CPPUNIT_TEST(runSummaryTest);
  CPPUNIT_TEST(lumiTest);
  CPPUNIT_TEST(lumiSummaryTest);
  CPPUNIT_TEST(beginRunProdTest);
  CPPUNIT_TEST(beginLumiProdTest);
  CPPUNIT_TEST(endRunProdTest);
  CPPUNIT_TEST(endLumiProdTest);
  CPPUNIT_TEST(endRunSummaryProdTest);
  CPPUNIT_TEST(endLumiSummaryProdTest);

  CPPUNIT_TEST_SUITE_END();

public:
  testGlobalProducer();

  void setUp() {}
  void tearDown() {}

  void basicTest();
  void streamTest();
  void runTest();
  void runSummaryTest();
  void lumiTest();
  void lumiSummaryTest();
  void beginRunProdTest();
  void beginLumiProdTest();
  void endRunProdTest();
  void endLumiProdTest();
  void endRunSummaryProdTest();
  void endLumiSummaryProdTest();

  enum class Trans {
    kBeginJob,
    kBeginStream,
    kGlobalBeginRun,
    kGlobalBeginRunProduce,
    kStreamBeginRun,
    kGlobalBeginLuminosityBlock,
    kStreamBeginLuminosityBlock,
    kEvent,
    kStreamEndLuminosityBlock,
    kGlobalEndLuminosityBlock,
    kStreamEndRun,
    kGlobalEndRun,
    kEndStream,
    kEndJob
  };
  typedef std::vector<Trans> Expectations;

private:
  std::map<Trans, std::function<void(edm::Worker*)>> m_transToFunc;

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
  edm::EventSetupImpl* m_es = nullptr;
  edm::ModuleDescription m_desc = {"Dummy", "dummy"};

  template <typename T>
  void testTransitions(std::shared_ptr<T> iMod, Expectations const& iExpect);

  template <typename Traits, typename Info>
  void doWork(edm::Worker* iBase, Info const& info, edm::ParentContext const& iContext) {
    oneapi::tbb::task_group group;
    edm::FinalWaitingTask task{group};
    edm::ServiceToken token;
    iBase->doWorkAsync<Traits>(edm::WaitingTaskHolder(group, &task), info, token, s_streamID0, iContext, nullptr);
    task.wait();
  }

  class BasicProd : public edm::global::EDProducer<> {
  public:
    mutable unsigned int m_count = 0;  //[[cms-thread-safe]]

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }
  };
  class StreamProd : public edm::global::EDProducer<edm::StreamCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::unique_ptr<int> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<int>{};
    }

    virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override { ++m_count; }
    virtual void streamBeginLuminosityBlock(edm::StreamID,
                                            edm::LuminosityBlock const&,
                                            edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndLuminosityBlock(edm::StreamID,
                                          edm::LuminosityBlock const&,
                                          edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override { ++m_count; }
    void endStream(edm::StreamID) const override { ++m_count; }
  };

  class RunProd : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override { ++m_count; }
  };

  class LumiProd : public edm::global::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                    edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override { ++m_count; }
  };

  class RunSummaryProd : public edm::global::EDProducer<edm::RunSummaryCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }

    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }
  };

  class LumiSummaryProd : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                           edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void streamEndLuminosityBlockSummary(edm::StreamID,
                                         edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         int*) const override {
      ++m_count;
    }

    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
  };

  class BeginRunProd : public edm::global::EDProducer<edm::BeginRunProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override { ++m_count; }
  };

  class BeginLumiProd : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndRunProd : public edm::global::EDProducer<edm::EndRunProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndLumiProd : public edm::global::EDProducer<edm::EndLuminosityBlockProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndRunSummaryProd : public edm::global::EDProducer<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    mutable bool m_globalEndRunSummaryCalled = false;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }

    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndRunSummaryCalled == false);
      m_globalEndRunSummaryCalled = true;
    }

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&, int const*) const override {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndRunSummaryCalled == true);
      m_globalEndRunSummaryCalled = false;
    }
  };

  class EndLumiSummaryProd
      : public edm::global::EDProducer<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    mutable bool m_globalEndLuminosityBlockSummaryCalled = false;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                           edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void streamEndLuminosityBlockSummary(edm::StreamID,
                                         edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         int*) const override {
      ++m_count;
    }

    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndLuminosityBlockSummaryCalled == false);
      m_globalEndLuminosityBlockSummaryCalled = true;
    }

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, int const*) const override {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndLuminosityBlockSummaryCalled == true);
      m_globalEndLuminosityBlockSummaryCalled = false;
    }
  };
  class TransformProd : public edm::global::EDProducer<edm::Transformer> {
  public:
    TransformProd(edm::ParameterSet const&) {
      token_ = produces<float>();
      registerTransform(token_, [](float iV) { return int(iV); });
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
      //iEvent.emplace(token_, 3.625);
    }

  private:
    edm::EDPutTokenT<float> token_;
  };

  class TransformAsyncProd : public edm::global::EDProducer<edm::Transformer> {
  public:
    struct IntHolder {
      IntHolder() : value_(0) {}
      IntHolder(int iV) : value_(iV) {}
      int value_;
    };
    TransformAsyncProd(edm::ParameterSet const&) {
      token_ = produces<float>();
      registerTransformAsync(
          token_,
          [](float iV, edm::WaitingTaskWithArenaHolder iHolder) { return IntHolder(iV); },
          [](IntHolder iWaitValue) { return iWaitValue.value_; });
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
      //iEvent.emplace(token_, 3.625);
    }

  private:
    edm::EDPutTokenT<float> token_;
  };
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testGlobalProducer);

testGlobalProducer::testGlobalProducer()
    : m_preallocConfig{},
      m_prodReg(new edm::ProductRegistry{}),
      m_idHelper(new edm::BranchIDListHelper{}),
      m_associationsHelper(new edm::ThinnedAssociationsHelper{}),
      m_ep() {
  //Setup the principals
  m_prodReg->setFrozen();
  m_idHelper->updateFromRegistry(*m_prodReg);
  edm::EventID eventID = edm::EventID::firstValidEvent();

  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp now(1234567UL);
  m_rp.reset(new edm::RunPrincipal(m_prodReg, m_procConfig, &historyAppender_, 0));
  m_rp->setAux(edm::RunAuxiliary(eventID.run(), now, now));
  auto lumiAux = std::make_shared<edm::LuminosityBlockAuxiliary>(m_rp->run(), 1, now, now);
  m_lbp.reset(new edm::LuminosityBlockPrincipal(m_prodReg, m_procConfig, &historyAppender_, 0));
  m_lbp->setAux(*lumiAux);
  m_lbp->setRunPrincipal(m_rp);
  edm::EventAuxiliary eventAux(eventID, uuid, now, true);

  m_ep.reset(new edm::EventPrincipal(m_prodReg, m_idHelper, m_associationsHelper, m_procConfig, nullptr));
  m_ep->fillEventPrincipal(eventAux, nullptr);
  m_ep->setLuminosityBlockPrincipal(m_lbp.get());
  m_actReg.reset(new edm::ActivityRegistry);

  //For each transition, bind a lambda which will call the proper method of the Worker
  m_transToFunc[Trans::kBeginStream] = [](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->beginStream(s_streamID0, streamContext);
  };

  m_transToFunc[Trans::kGlobalBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::GlobalContext gc(edm::GlobalContext::Transition::kBeginRun, nullptr);
    edm::ParentContext nullParentContext(&gc);
    iBase->setActivityRegistry(m_actReg);
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::GlobalContext gc(edm::GlobalContext::Transition::kBeginLuminosityBlock, nullptr);
    edm::ParentContext nullParentContext(&gc);
    iBase->setActivityRegistry(m_actReg);
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kEvent] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    edm::EventTransitionInfo info(*m_ep, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kStreamEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::GlobalContext gc(edm::GlobalContext::Transition::kEndLuminosityBlock, nullptr);
    edm::ParentContext nullParentContext(&gc);
    iBase->setActivityRegistry(m_actReg);
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::GlobalContext gc(edm::GlobalContext::Transition::kEndRun, nullptr);
    edm::ParentContext nullParentContext(&gc);
    iBase->setActivityRegistry(m_actReg);
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kEndStream] = [](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->endStream(s_streamID0, streamContext);
  };
}

namespace {
  template <typename T>
  void testTransition(std::shared_ptr<T> iMod,
                      edm::Worker* iWorker,
                      testGlobalProducer::Trans iTrans,
                      testGlobalProducer::Expectations const& iExpect,
                      std::function<void(edm::Worker*)> iFunc) {
    assert(0 == iMod->m_count);
    iFunc(iWorker);
    auto count = std::count(iExpect.begin(), iExpect.end(), iTrans);
    if (count != iMod->m_count) {
      std::cout << "For trans " << static_cast<std::underlying_type<testGlobalProducer::Trans>::type>(iTrans)
                << " expected " << count << " and got " << iMod->m_count << std::endl;
    }
    CPPUNIT_ASSERT(iMod->m_count == count);
    iMod->m_count = 0;
    iWorker->reset();
  }
}  // namespace

template <typename T>
void testGlobalProducer::testTransitions(std::shared_ptr<T> iMod, Expectations const& iExpect) {
  oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  oneapi::tbb::task_arena arena(1);
  arena.execute([&]() {
    edm::maker::ModuleHolderT<edm::global::EDProducerBase> h(iMod, nullptr);
    h.preallocate(edm::PreallocationConfiguration{});

    edm::WorkerT<edm::global::EDProducerBase> w{iMod, m_desc, nullptr};
    for (auto& keyVal : m_transToFunc) {
      testTransition(iMod, &w, keyVal.first, iExpect, keyVal.second);
    }
  });
}

void testGlobalProducer::basicTest() {
  auto testProd = std::make_shared<BasicProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kEvent});
}

void testGlobalProducer::streamTest() {
  auto testProd = std::make_shared<StreamProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd,
                  {Trans::kBeginStream,
                   Trans::kStreamBeginRun,
                   Trans::kStreamBeginLuminosityBlock,
                   Trans::kEvent,
                   Trans::kStreamEndLuminosityBlock,
                   Trans::kStreamEndRun,
                   Trans::kEndStream});
}

void testGlobalProducer::runTest() {
  auto testProd = std::make_shared<RunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun});
}

void testGlobalProducer::runSummaryTest() {
  auto testProd = std::make_shared<RunSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testGlobalProducer::lumiTest() {
  auto testProd = std::make_shared<LumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock});
}

void testGlobalProducer::lumiSummaryTest() {
  auto testProd = std::make_shared<LumiSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd,
                  {Trans::kGlobalBeginLuminosityBlock,
                   Trans::kEvent,
                   Trans::kStreamEndLuminosityBlock,
                   Trans::kGlobalEndLuminosityBlock});
}

void testGlobalProducer::beginRunProdTest() {
  auto testProd = std::make_shared<BeginRunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent});
}

void testGlobalProducer::beginLumiProdTest() {
  auto testProd = std::make_shared<BeginLumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent});
}

void testGlobalProducer::endRunProdTest() {
  auto testProd = std::make_shared<EndRunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalEndRun, Trans::kEvent});
}

void testGlobalProducer::endLumiProdTest() {
  auto testProd = std::make_shared<EndLumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalEndLuminosityBlock, Trans::kEvent});
}

void testGlobalProducer::endRunSummaryProdTest() {
  auto testProd = std::make_shared<EndRunSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(
      testProd,
      {Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testGlobalProducer::endLumiSummaryProdTest() {
  auto testProd = std::make_shared<EndLumiSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd,
                  {Trans::kGlobalEndLuminosityBlock,
                   Trans::kEvent,
                   Trans::kGlobalBeginLuminosityBlock,
                   Trans::kStreamEndLuminosityBlock,
                   Trans::kGlobalEndLuminosityBlock});
}
