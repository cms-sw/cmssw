/*
 *  stream_filter_test.cppunit.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 2/8/2013.
 */
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/stream/EDProducerAdaptor.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

class testStreamFilter : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testStreamFilter);

  CPPUNIT_TEST(basicTest);
  CPPUNIT_TEST(globalTest);
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
  testStreamFilter();

  void setUp() {}
  void tearDown() {}

  void basicTest();
  void globalTest();
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
    kBeginJob,                    //0
    kBeginStream,                 //1
    kGlobalBeginRun,              //2
    kStreamBeginRun,              //3
    kGlobalBeginLuminosityBlock,  //4
    kStreamBeginLuminosityBlock,  //5
    kEvent,                       //6
    kStreamEndLuminosityBlock,    //7
    kGlobalEndLuminosityBlock,    //8
    kStreamEndRun,                //9
    kGlobalEndRun,                //10
    kEndStream,                   //11
    kEndJob                       //12
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

  template <typename T, typename U>
  void testTransitions(std::shared_ptr<U> iMod, Expectations const& iExpect);

  template <typename T>
  void runTest(Expectations const& iExpect);

  class BasicProd : public edm::stream::EDFilter<> {
  public:
    static unsigned int m_count;

    BasicProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }
  };

  class GlobalProd : public edm::stream::EDFilter<edm::GlobalCache<int>> {
  public:
    static unsigned int m_count;

    static std::unique_ptr<int> initializeGlobalCache(edm::ParameterSet const&) { return std::make_unique<int>(1); }
    GlobalProd(edm::ParameterSet const&, const int* iGlobal) { CPPUNIT_ASSERT(*iGlobal == 1); }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalEndJob(int* iGlobal) {
      CPPUNIT_ASSERT(1 == *iGlobal);
      ++m_count;
    }
  };
  class RunProd : public edm::stream::EDFilter<edm::RunCache<int>> {
  public:
    static unsigned int m_count;
    RunProd(edm::ParameterSet const&) {}
    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    static void globalEndRun(edm::Run const&, edm::EventSetup const&, RunProd::RunContext const*) { ++m_count; }
  };

  class LumiProd : public edm::stream::EDFilter<edm::LuminosityBlockCache<int>> {
  public:
    static unsigned int m_count;
    LumiProd(edm::ParameterSet const&) {}
    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                           edm::EventSetup const&,
                                                           RunContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    static void globalEndLuminosityBlock(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         LuminosityBlockContext const*) {
      ++m_count;
    }
  };

  class RunSummaryProd : public edm::stream::EDFilter<edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count;
    RunSummaryProd(edm::ParameterSet const&) {}
    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void endRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }

    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, int*) { ++m_count; }
  };

  class LumiSummaryProd : public edm::stream::EDFilter<edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count;
    LumiSummaryProd(edm::ParameterSet const&) {}
    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                  edm::EventSetup const&,
                                                                  LuminosityBlockContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }

    static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                edm::EventSetup const&,
                                                LuminosityBlockContext const*,
                                                int*) {
      ++m_count;
    }
  };

  class BeginRunProd : public edm::stream::EDFilter<edm::BeginRunProducer> {
  public:
    static unsigned int m_count;
    BeginRunProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalBeginRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*) { ++m_count; }
  };

  class BeginLumiProd : public edm::stream::EDFilter<edm::BeginLuminosityBlockProducer> {
  public:
    static unsigned int m_count;
    BeginLumiProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const*) {
      ++m_count;
    }
  };

  class EndRunProd : public edm::stream::EDFilter<edm::EndRunProducer> {
  public:
    static unsigned int m_count;
    EndRunProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*) { ++m_count; }
  };

  class EndLumiProd : public edm::stream::EDFilter<edm::EndLuminosityBlockProducer> {
  public:
    static unsigned int m_count;
    EndLumiProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                                edm::EventSetup const&,
                                                LuminosityBlockContext const*) {
      ++m_count;
    }
  };

  class EndRunSummaryProd : public edm::stream::EDFilter<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count;
    static bool m_globalEndRunSummaryCalled;
    EndRunSummaryProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void endRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override { ++m_count; }

    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, int*) {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndRunSummaryCalled == false);
      m_globalEndRunSummaryCalled = true;
    }

    static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*, int const*) {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndRunSummaryCalled == true);
      m_globalEndRunSummaryCalled = false;
    }
  };

  class EndLumiSummaryProd
      : public edm::stream::EDFilter<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count;
    static bool m_globalEndLuminosityBlockSummaryCalled;
    EndLumiSummaryProd(edm::ParameterSet const&) {}

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                  edm::EventSetup const&,
                                                                  LuminosityBlockContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }

    static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                edm::EventSetup const&,
                                                LuminosityBlockContext const*,
                                                int*) {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndLuminosityBlockSummaryCalled == false);
      m_globalEndLuminosityBlockSummaryCalled = true;
    }

    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                                edm::EventSetup const&,
                                                LuminosityBlockContext const*,
                                                int const*) {
      ++m_count;
      CPPUNIT_ASSERT(m_globalEndLuminosityBlockSummaryCalled == true);
      m_globalEndLuminosityBlockSummaryCalled = false;
    }
  };
};
unsigned int testStreamFilter::BasicProd::m_count = 0;
unsigned int testStreamFilter::GlobalProd::m_count = 0;
unsigned int testStreamFilter::RunProd::m_count = 0;
unsigned int testStreamFilter::LumiProd::m_count = 0;
unsigned int testStreamFilter::RunSummaryProd::m_count = 0;
unsigned int testStreamFilter::LumiSummaryProd::m_count = 0;
unsigned int testStreamFilter::BeginRunProd::m_count = 0;
unsigned int testStreamFilter::EndRunProd::m_count = 0;
unsigned int testStreamFilter::BeginLumiProd::m_count = 0;
unsigned int testStreamFilter::EndLumiProd::m_count = 0;
unsigned int testStreamFilter::EndRunSummaryProd::m_count = 0;
bool testStreamFilter::EndRunSummaryProd::m_globalEndRunSummaryCalled = false;
unsigned int testStreamFilter::EndLumiSummaryProd::m_count = 0;
bool testStreamFilter::EndLumiSummaryProd::m_globalEndLuminosityBlockSummaryCalled = false;
///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testStreamFilter);

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

testStreamFilter::testStreamFilter()
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
  auto lumiAux = std::make_shared<edm::LuminosityBlockAuxiliary>(m_rp->run(), 1, now, now);
  m_lbp.reset(new edm::LuminosityBlockPrincipal(m_prodReg, m_procConfig, &historyAppender_, 0));
  m_lbp->setAux(*lumiAux);
  m_lbp->setRunPrincipal(m_rp);
  edm::EventAuxiliary eventAux(eventID, uuid, now, true);

  //Only an EventProcessor or SubProcess is allowed to create a StreamID but I need one
  ShadowStreamID shadowID;
  shadowID.value = 0;
  edm::StreamID* pID = reinterpret_cast<edm::StreamID*>(&shadowID);
  assert(pID->value() == 0);

  m_ep.reset(new edm::EventPrincipal(m_prodReg, m_idHelper, m_associationsHelper, m_procConfig, nullptr, *pID));
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
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp, *m_es, s_streamID0, parentContext, nullptr);
  };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp, *m_es, s_streamID0, parentContext, nullptr);
  };

  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp, *m_es, s_streamID0, parentContext, nullptr);
  };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp, *m_es, s_streamID0, parentContext, nullptr);
  };

  m_transToFunc[Trans::kEvent] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext parentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    iBase->doWork<Traits>(*m_ep, *m_es, s_streamID0, parentContext, nullptr);
  };

  m_transToFunc[Trans::kStreamEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp, *m_es, s_streamID0, parentContext, nullptr);
  };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp, *m_es, s_streamID0, parentContext, nullptr);
  };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp, *m_es, s_streamID0, parentContext, nullptr);
  };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp, *m_es, s_streamID0, parentContext, nullptr);
  };

  m_transToFunc[Trans::kEndStream] = [](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->endStream(s_streamID0, streamContext);
  };
}

namespace {
  template <typename T>
  std::shared_ptr<edm::stream::EDFilterAdaptorBase> createModule() {
    edm::ParameterSet pset;
    std::shared_ptr<edm::stream::EDFilterAdaptorBase> retValue =
        std::make_shared<edm::stream::EDFilterAdaptor<T>>(pset);
    edm::maker::ModuleHolderT<edm::stream::EDFilterAdaptorBase> h(retValue, nullptr);
    h.preallocate(edm::PreallocationConfiguration{});
    return retValue;
  }
  template <typename T>
  void testTransition(edm::Worker* iWorker,
                      testStreamFilter::Trans iTrans,
                      testStreamFilter::Expectations const& iExpect,
                      std::function<void(edm::Worker*)> iFunc) {
    assert(0 == T::m_count);
    iFunc(iWorker);
    auto count = std::count(iExpect.begin(), iExpect.end(), iTrans);
    if (count != T::m_count) {
      std::cout << "For trans " << static_cast<std::underlying_type<testStreamFilter::Trans>::type>(iTrans)
                << " expected " << count << " and got " << T::m_count << std::endl;
    }
    CPPUNIT_ASSERT(T::m_count == count);
    T::m_count = 0;
    iWorker->reset();
  }
}  // namespace

template <typename T, typename U>
void testStreamFilter::testTransitions(std::shared_ptr<U> iMod, Expectations const& iExpect) {
  edm::WorkerT<edm::stream::EDFilterAdaptorBase> w{iMod, m_desc, nullptr};
  for (auto& keyVal : m_transToFunc) {
    testTransition<T>(&w, keyVal.first, iExpect, keyVal.second);
  }
}
template <typename T>
void testStreamFilter::runTest(Expectations const& iExpect) {
  auto mod = createModule<T>();
  CPPUNIT_ASSERT(0 == T::m_count);
  testTransitions<T>(mod, iExpect);
}

void testStreamFilter::basicTest() { runTest<BasicProd>({Trans::kEvent}); }

void testStreamFilter::globalTest() { runTest<GlobalProd>({Trans::kBeginJob, Trans::kEvent, Trans::kEndJob}); }

void testStreamFilter::runTest() { runTest<RunProd>({Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun}); }

void testStreamFilter::runSummaryTest() {
  runTest<RunSummaryProd>({Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testStreamFilter::lumiTest() {
  runTest<LumiProd>({Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock});
}

void testStreamFilter::lumiSummaryTest() {
  runTest<LumiSummaryProd>({Trans::kGlobalBeginLuminosityBlock,
                            Trans::kEvent,
                            Trans::kStreamEndLuminosityBlock,
                            Trans::kGlobalEndLuminosityBlock});
}

void testStreamFilter::beginRunProdTest() { runTest<BeginRunProd>({Trans::kGlobalBeginRun, Trans::kEvent}); }

void testStreamFilter::beginLumiProdTest() {
  runTest<BeginLumiProd>({Trans::kGlobalBeginLuminosityBlock, Trans::kEvent});
}

void testStreamFilter::endRunProdTest() { runTest<EndRunProd>({Trans::kGlobalEndRun, Trans::kEvent}); }

void testStreamFilter::endLumiProdTest() { runTest<EndLumiProd>({Trans::kGlobalEndLuminosityBlock, Trans::kEvent}); }

void testStreamFilter::endRunSummaryProdTest() {
  runTest<EndRunSummaryProd>(
      {Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testStreamFilter::endLumiSummaryProdTest() {
  runTest<EndLumiSummaryProd>({Trans::kGlobalEndLuminosityBlock,
                               Trans::kEvent,
                               Trans::kGlobalBeginLuminosityBlock,
                               Trans::kStreamEndLuminosityBlock,
                               Trans::kGlobalEndLuminosityBlock});
}
