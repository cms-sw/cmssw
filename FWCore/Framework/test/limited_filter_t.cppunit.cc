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
#include "tbb/global_control.h"
#include "FWCore/Framework/interface/limited/EDFilter.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

namespace {
  edm::ParameterSet makePSet() {
    edm::ParameterSet pset;
    const unsigned int kLimit = 1;
    pset.addUntrackedParameter("concurrencyLimit", kLimit);
    return pset;
  }

  const edm::ParameterSet s_pset = makePSet();
}  // namespace

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

class testLimitedFilter : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testLimitedFilter);

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
  testLimitedFilter();

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
    auto task = edm::make_empty_waiting_task();
    task->increment_ref_count();
    iBase->doWorkAsync<Traits>(
        edm::WaitingTaskHolder(task.get()), info, edm::ServiceToken(), s_streamID0, iContext, nullptr);
    task->wait_for_all();
    if (auto e = task->exceptionPtr()) {
      std::rethrow_exception(*e);
    }
  }

  class BasicProd : public edm::limited::EDFilter<> {
  public:
    BasicProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<>(s_pset) {}
    mutable unsigned int m_count = 0;  //[[cms-thread-safe]]

    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
  };
  class StreamProd : public edm::limited::EDFilter<edm::StreamCache<int>> {
  public:
    StreamProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::StreamCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

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

  class RunProd : public edm::limited::EDFilter<edm::RunCache<int>> {
  public:
    RunProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::RunCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override { ++m_count; }
  };

  class LumiProd : public edm::limited::EDFilter<edm::LuminosityBlockCache<int>> {
  public:
    LumiProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::LuminosityBlockCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                    edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override { ++m_count; }
  };

  class RunSummaryProd : public edm::limited::EDFilter<edm::RunSummaryCache<int>> {
  public:
    RunSummaryProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::RunSummaryCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }

    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }
  };

  class LumiSummaryProd : public edm::limited::EDFilter<edm::LuminosityBlockSummaryCache<int>> {
  public:
    LumiSummaryProd()
        : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::LuminosityBlockSummaryCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

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

  class BeginRunProd : public edm::limited::EDFilter<edm::BeginRunProducer> {
  public:
    BeginRunProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::BeginRunProducer>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override { ++m_count; }
  };

  class BeginLumiProd : public edm::limited::EDFilter<edm::BeginLuminosityBlockProducer> {
  public:
    BeginLumiProd()
        : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::BeginLuminosityBlockProducer>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndRunProd : public edm::limited::EDFilter<edm::EndRunProducer> {
  public:
    EndRunProd() : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::EndRunProducer>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndLumiProd : public edm::limited::EDFilter<edm::EndLuminosityBlockProducer> {
  public:
    EndLumiProd()
        : edm::limited::EDFilterBase(s_pset), edm::limited::EDFilter<edm::EndLuminosityBlockProducer>(s_pset) {}
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override { ++m_count; }
  };

  class EndRunSummaryProd : public edm::limited::EDFilter<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    EndRunSummaryProd()
        : edm::limited::EDFilterBase(s_pset),
          edm::limited::EDFilter<edm::EndRunProducer, edm::RunSummaryCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    mutable bool m_globalEndRunSummaryCalled = false;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

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
      : public edm::limited::EDFilter<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>> {
  public:
    EndLumiSummaryProd()
        : edm::limited::EDFilterBase(s_pset),
          edm::limited::EDFilter<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>>(s_pset) {}
    mutable unsigned int m_count = 0;
    mutable bool m_globalEndLuminosityBlockSummaryCalled = false;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }

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
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testLimitedFilter);

testLimitedFilter::testLimitedFilter()
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
  auto runAux = std::make_shared<edm::RunAuxiliary>(eventID.run(), now, now);
  m_rp.reset(new edm::RunPrincipal(runAux, m_prodReg, m_procConfig, &historyAppender_, 0));
  edm::LuminosityBlockAuxiliary lumiAux(m_rp->run(), 1, now, now);
  m_lbp.reset(new edm::LuminosityBlockPrincipal(m_prodReg, m_procConfig, &historyAppender_, 0));
  m_lbp->setAux(lumiAux);
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
    edm::ParentContext nullParentContext;
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext nullParentContext;
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext nullParentContext;
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext nullParentContext;
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
    edm::ParentContext nullParentContext;
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext nullParentContext;
    edm::LumiTransitionInfo info(*m_lbp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext nullParentContext;
    edm::RunTransitionInfo info(*m_rp, *m_es);
    doWork<Traits>(iBase, info, nullParentContext);
  };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext nullParentContext;
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
                      testLimitedFilter::Trans iTrans,
                      testLimitedFilter::Expectations const& iExpect,
                      std::function<void(edm::Worker*)> iFunc) {
    assert(0 == iMod->m_count);
    iFunc(iWorker);
    auto count = std::count(iExpect.begin(), iExpect.end(), iTrans);
    if (count != iMod->m_count) {
      std::cout << "For trans " << static_cast<std::underlying_type<testLimitedFilter::Trans>::type>(iTrans)
                << " expected " << count << " and got " << iMod->m_count << std::endl;
    }
    CPPUNIT_ASSERT(iMod->m_count == count);
    iMod->m_count = 0;
    iWorker->reset();
  }
}  // namespace

template <typename T>
void testLimitedFilter::testTransitions(std::shared_ptr<T> iMod, Expectations const& iExpect) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  edm::maker::ModuleHolderT<edm::limited::EDFilterBase> h(iMod, nullptr);
  h.preallocate(edm::PreallocationConfiguration{});
  edm::WorkerT<edm::limited::EDFilterBase> w{iMod, m_desc, nullptr};
  for (auto& keyVal : m_transToFunc) {
    testTransition(iMod, &w, keyVal.first, iExpect, keyVal.second);
  }
}

void testLimitedFilter::basicTest() {
  auto testProd = std::make_shared<BasicProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kEvent});
}

void testLimitedFilter::streamTest() {
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

void testLimitedFilter::runTest() {
  auto testProd = std::make_shared<RunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun});
}

void testLimitedFilter::runSummaryTest() {
  auto testProd = std::make_shared<RunSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testLimitedFilter::lumiTest() {
  auto testProd = std::make_shared<LumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock});
}

void testLimitedFilter::lumiSummaryTest() {
  auto testProd = std::make_shared<LumiSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd,
                  {Trans::kGlobalBeginLuminosityBlock,
                   Trans::kEvent,
                   Trans::kStreamEndLuminosityBlock,
                   Trans::kGlobalEndLuminosityBlock});
}

void testLimitedFilter::beginRunProdTest() {
  auto testProd = std::make_shared<BeginRunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginRun, Trans::kEvent});
}

void testLimitedFilter::beginLumiProdTest() {
  auto testProd = std::make_shared<BeginLumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent});
}

void testLimitedFilter::endRunProdTest() {
  auto testProd = std::make_shared<EndRunProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalEndRun, Trans::kEvent});
}

void testLimitedFilter::endLumiProdTest() {
  auto testProd = std::make_shared<EndLumiProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd, {Trans::kGlobalEndLuminosityBlock, Trans::kEvent});
}

void testLimitedFilter::endRunSummaryProdTest() {
  auto testProd = std::make_shared<EndRunSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(
      testProd,
      {Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testLimitedFilter::endLumiSummaryProdTest() {
  auto testProd = std::make_shared<EndLumiSummaryProd>();

  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd,
                  {Trans::kGlobalEndLuminosityBlock,
                   Trans::kEvent,
                   Trans::kGlobalBeginLuminosityBlock,
                   Trans::kStreamEndLuminosityBlock,
                   Trans::kGlobalEndLuminosityBlock});
}
