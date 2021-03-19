
/*----------------------------------------------------------------------

Toy edm::stream::EDProducer modules of
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {
  namespace stream {

    // anonymous namespace here causes build warnings
    namespace cache {
      struct Cache {
        Cache() : value(0), run(0), lumi(0) {}
        //Using mutable since we want to update the value.
        mutable std::atomic<unsigned int> value;
        mutable std::atomic<unsigned int> run;
        mutable std::atomic<unsigned int> lumi;
      };

      struct UnsafeCache {
        UnsafeCache() : value(0), run(0), lumi(0) {}
        unsigned int value;
        unsigned int run;
        unsigned int lumi;
      };

      struct TestGlobalCache {
        CMS_THREAD_SAFE mutable edm::EDPutTokenT<unsigned int> token_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenBegin_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenEnd_;
        unsigned int trans_{0};
        CMS_THREAD_SAFE mutable std::atomic<unsigned int> m_count{0};
      };

    }  // namespace cache

    using Cache = cache::Cache;
    using UnsafeCache = cache::UnsafeCache;
    using TestGlobalCache = cache::TestGlobalCache;

    class GlobalIntProducer : public edm::stream::EDProducer<edm::GlobalCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const&) {
        ++m_count;
        return std::make_unique<Cache>();
      }

      GlobalIntProducer(edm::ParameterSet const& p, const Cache* iGlobal) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      static void globalBeginJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != 0) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be 0";
        }
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++((globalCache())->value);
      }

      static void globalEndJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != cvalue_) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~GlobalIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntProducer : public edm::stream::EDProducer<edm::RunCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      bool br;
      bool er;

      RunIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
      }

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        gbr = true;
        ger = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->run);
        return pCache;
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        br = true;
        er = true;
        if (!gbr) {
          throw cms::Exception("begin out of sequence") << "beginRun seen before globalBeginRun";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        auto pCache = iContext->run();
        if (pCache->run != 1) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        ger = true;
        gbr = false;
        if (iContext->run()->value != cvalue_) {
          throw cms::Exception("cache value") << iContext->run()->value << " but it was supposed to be " << cvalue_;
        }
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {
        er = true;
        br = false;
        if (ger) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before endRun";
        }
      }

      ~RunIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> bl;
      static std::atomic<bool> el;

      LumiIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        gbl = true;
        gel = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->lumi);
        return pCache;
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        bl = true;
        el = false;
        if (!gbl) {
          throw cms::Exception("begin out of sequence")
              << "beginLuminosityBlock seen before globalBeginLuminosityBlock";
        }
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->lumi != 1) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
        if (iLBContext->luminosityBlock()->value != cvalue_) {
          throw cms::Exception("cache value")
              << iLBContext->luminosityBlock()->value << " but it was supposed to be " << cvalue_;
        }
      }

      static void endLuminosityBlock(edm::Run const&, edm::EventSetup const&, LuminosityBlockContext const*) {
        el = true;
        bl = false;
        if (gel) {
          throw cms::Exception("end out of sequence") << "globalEndLuminosityBlock seen before endLuminosityBlock";
        }
      }

      ~LumiIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntProducer
        : public edm::stream::EDProducer<edm::RunCache<Cache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      static std::atomic<bool> gbrs;
      static std::atomic<bool> gers;
      static std::atomic<bool> brs;
      static std::atomic<bool> ers;
      static std::atomic<bool> br;
      static std::atomic<bool> er;

      RunSummaryIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        br = true;
        er = false;
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
      }

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        gbr = true;
        ger = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->run);
        return pCache;
      }

      static std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const&,
                                                                edm::EventSetup const&,
                                                                GlobalCache const*) {
        ++m_count;
        gbrs = true;
        gers = false;
        brs = true;
        ers = false;
        if (!gbr) {
          throw cms::Exception("begin out of sequence") << "globalBeginRunSummary seen before globalBeginRun";
        }
        return std::make_shared<UnsafeCache>();
      }

      void endRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
        brs = false;
        ers = true;
        gCache->value += runCache()->value;
        runCache()->value = 0;
        if (!er) {
          throw cms::Exception("end out of sequence") << "endRunSummary seen before endRun";
        }
      }

      static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, UnsafeCache* gCache) {
        ++m_count;
        gbrs = false;
        gers = true;
        if (!ers) {
          throw cms::Exception("end out of sequence") << "globalEndRunSummary seen before endRunSummary";
        }
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value") << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        gbr = false;
        ger = true;
        auto pCache = iContext->run();
        if (pCache->run != 1) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {
        er = true;
        br = false;
      }

      ~RunSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>,
                                                                  edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      static std::atomic<unsigned int> m_lumiSumCalls;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> gbls;
      static std::atomic<bool> gels;
      static std::atomic<bool> bls;
      static std::atomic<bool> els;
      static std::atomic<bool> bl;
      static std::atomic<bool> el;

      LumiSummaryIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        bl = true;
        el = false;
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        gbl = true;
        gel = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->lumi);
        return pCache;
      }

      static std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                            edm::EventSetup const&,
                                                                            LuminosityBlockContext const*) {
        ++m_count;
        gbls = true;
        gels = false;
        bls = true;
        els = false;
        if (!gbl) {
          throw cms::Exception("begin out of sequence")
              << "globalBeginLuminosityBlockSummary seen before globalBeginLuminosityBlock";
        }
        return std::make_shared<UnsafeCache>();
      }

      void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                     edm::EventSetup const&,
                                     UnsafeCache* gCache) const override {
        ++m_lumiSumCalls;
        bls = false;
        els = true;
        //This routine could be called at the same time as another stream is calling produce so must do the change atomically
        auto v = luminosityBlockCache()->value.exchange(0);
        gCache->value += v;
        if (el) {
          throw cms::Exception("end out of sequence") << "endLuminosityBlock seen before endLuminosityBlockSummary";
        }
      }

      static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const*,
                                                  UnsafeCache* gCache) {
        ++m_count;
        auto nLumis = m_lumiSumCalls.load();
        gbls = false;
        gels = true;
        if (!els) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlockSummary seen before endLuminosityBlockSummary";
        }
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << gCache->value << " but it was supposed to be " << cvalue_ << " endLumiBlockSummary called " << nLumis;
        }
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->lumi != 1) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
        if (!gels) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlockSummary seen before globalEndLuminosityBlock";
        }
      }

      static void endLuminosityBlock(edm::Run const&, edm::EventSetup const&, LuminosityBlockContext const*) {
        el = true;
        bl = false;
      }

      ~LumiSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntProducer
        : public edm::stream::EDProducer<edm::WatchProcessBlock, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit ProcessBlockIntProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            testGlobalCache->getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            testGlobalCache->getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer::begin transitions "
                                              << testGlobalCache->m_count << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;

        const unsigned int valueToGet = 51;
        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
      }

      static void endProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCache* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer::end transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
        {
          const unsigned int valueToGet = 51;
          if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 61;
          if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
            }
          }
        }
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~ProcessBlockIntProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestBeginProcessBlockProducer
        : public edm::stream::EDProducer<edm::BeginProcessBlockProducer, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit TestBeginProcessBlockProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::BeginProcessBlock>("begin");

        auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCache const* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "TestBeginProcessBlockProducer transitions "
                                              << testGlobalCache->m_count << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;

        const unsigned int valueToPutAndGet = 51;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);

        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence")
              << "produce before beginProcessBlockProduce " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestBeginProcessBlockProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestEndProcessBlockProducer
        : public edm::stream::EDProducer<edm::EndProcessBlockProducer, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit TestEndProcessBlockProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::EndProcessBlock>("end");

        auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        ++testGlobalCache->m_count;
      }

      static void endProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCache const* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestEndProcessBlockProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
        const unsigned int valueToPutAndGet = 61;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);

        if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
          }
        }
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestEndProcessBlockProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "~TestEndProcessBlockProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class ProcessBlockIntProducerNoGlobalCache : public edm::stream::EDProducer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntProducerNoGlobalCache(edm::ParameterSet const&) {}

      static void beginProcessBlock(edm::ProcessBlock const&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}

      static void endProcessBlock(edm::ProcessBlock const&) {}
    };

    class TestBeginProcessBlockProducerNoGlobalCache : public edm::stream::EDProducer<edm::BeginProcessBlockProducer> {
    public:
      explicit TestBeginProcessBlockProducerNoGlobalCache(edm::ParameterSet const&) {}

      static void beginProcessBlockProduce(edm::ProcessBlock&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}
    };

    class TestEndProcessBlockProducerNoGlobalCache : public edm::stream::EDProducer<edm::EndProcessBlockProducer> {
    public:
      explicit TestEndProcessBlockProducerNoGlobalCache(edm::ParameterSet const&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}

      static void endProcessBlockProduce(edm::ProcessBlock&) {}
    };

    class TestBeginRunProducer : public edm::stream::EDProducer<edm::RunCache<bool>, edm::BeginRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      static std::atomic<bool> gbrp;

      TestBeginRunProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }

      static std::shared_ptr<bool> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        gbr = true;
        ger = false;
        gbrp = false;
        return std::shared_ptr<bool>{};
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        if (!gbrp) {
          throw cms::Exception("out of sequence") << "produce before globalBeginRunProduce";
        }
      }

      static void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        gbrp = true;
        ++m_count;
        if (!gbr) {
          throw cms::Exception("begin out of sequence") << "globalBeginRunProduce seen before globalBeginRun";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        if (!gbr) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestBeginRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndRunProducer : public edm::stream::EDProducer<edm::RunCache<bool>, edm::EndRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      static std::atomic<bool> p;

      static std::shared_ptr<bool> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        gbr = true;
        ger = false;
        p = false;
        return std::shared_ptr<bool>{};
      }

      TestEndRunProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndRun>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override { p = true; }

      static void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        ++m_count;
        if (!p) {
          throw cms::Exception("out of sequence") << "globalEndRunProduce seen before produce";
        }
        if (ger) {
          throw cms::Exception("out of sequence") << "globalEndRun seen before globalEndRunProduce";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        if (!gbr) {
          throw cms::Exception("out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestEndRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginLumiBlockProducer
        : public edm::stream::EDProducer<edm::LuminosityBlockCache<bool>, edm::BeginLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> gblp;

      TestBeginLumiBlockProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        if (!gblp) {
          throw cms::Exception("begin out of sequence") << "produce seen before globalBeginLumiBlockProduce";
        }
      }

      static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&,
                                                    edm::EventSetup const&,
                                                    LuminosityBlockContext const*) {
        ++m_count;
        if (!gbl) {
          throw cms::Exception("begin out of sequence")
              << "globalBeginLumiBlockProduce seen before globalBeginLumiBlock";
        }
        gblp = true;
      }

      static std::shared_ptr<bool> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                              edm::EventSetup const&,
                                                              RunContext const*) {
        gbl = true;
        gel = false;
        gblp = false;
        return std::shared_ptr<bool>();
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        if (!gbl) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
      }

      ~TestBeginLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndLumiBlockProducer
        : public edm::stream::EDProducer<edm::LuminosityBlockCache<bool>, edm::EndLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> p;

      TestEndLumiBlockProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override { p = true; }

      static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const*) {
        ++m_count;
        if (!p) {
          throw cms::Exception("out of sequence") << "globalEndLumiBlockProduce seen before produce";
        }
      }

      static std::shared_ptr<bool> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                              edm::EventSetup const&,
                                                              RunContext const*) {
        gbl = true;
        gel = false;
        p = false;
        return std::shared_ptr<bool>{};
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        if (!gbl) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
      }

      ~TestEndLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    struct Count {
      Count() : m_value(0), m_expectedValue(0) {}
      //Using mutable since we want to update the value.
      mutable std::atomic<unsigned int> m_value;
      mutable std::atomic<unsigned int> m_expectedValue;
      // Set only in constructor, framework runs constructors serially
      CMS_THREAD_SAFE mutable edm::EDPutTokenT<unsigned int> m_putToken;
    };

    class TestAccumulator
        : public edm::stream::EDProducer<edm::GlobalCache<Count>, edm::Accumulator, edm::EndLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_expectedCount;

      explicit TestAccumulator(edm::ParameterSet const& p, Count const* iCount) {
        iCount->m_expectedValue = p.getParameter<unsigned int>("expectedCount");
        if (iCount->m_putToken.isUninitialized()) {
          iCount->m_putToken = produces<unsigned int, edm::Transition::EndLuminosityBlock>();
        }
      }

      static std::unique_ptr<Count> initializeGlobalCache(edm::ParameterSet const&) {
        return std::unique_ptr<Count>(new Count());
      }

      void accumulate(edm::Event const&, edm::EventSetup const&) override { ++(globalCache()->m_value); }

      static void globalEndLuminosityBlockProduce(edm::LuminosityBlock& l,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const* ctx) {
        Count const* count = ctx->global();
        l.emplace(count->m_putToken, count->m_value.load());
      }

      static void globalEndJob(Count const* iCount) {
        if (iCount->m_value != iCount->m_expectedValue) {
          throw cms::Exception("CountEvents") << "Number of events seen = " << iCount->m_value
                                              << " but it was supposed to be " << iCount->m_expectedValue;
        }
      }

      ~TestAccumulator() {}
    };

  }  // namespace stream
}  // namespace edmtest
std::atomic<unsigned int> edmtest::stream::GlobalIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockProducer::cvalue_{0};
std::atomic<bool> edmtest::stream::RunIntProducer::gbr{false};
std::atomic<bool> edmtest::stream::RunIntProducer::ger{false};
std::atomic<bool> edmtest::stream::LumiIntProducer::gbl{false};
std::atomic<bool> edmtest::stream::LumiIntProducer::gel{false};
std::atomic<bool> edmtest::stream::LumiIntProducer::bl{false};
std::atomic<bool> edmtest::stream::LumiIntProducer::el{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::gbr{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::ger{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::gbrs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::gers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::brs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::ers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::br{false};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::er{false};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::m_lumiSumCalls{0};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::gbl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::gel{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::gbls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::gels{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::bls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::els{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::bl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::el{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::gbr{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::gbrp{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::gbr{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::p{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gbl{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gblp{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::gbl{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::p{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginProcessBlockProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestEndProcessBlockProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestAccumulator);
