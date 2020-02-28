
/*----------------------------------------------------------------------

Toy edm::limited::EDProducer modules of 
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/limited/EDProducer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {
  namespace limited {

    namespace {
      struct Cache {
        Cache() : value(0), run(0), lumi(0), strm(0), work(0) {}
        //Using mutable since we want to update the value.
        mutable std::atomic<unsigned int> value;
        mutable std::atomic<unsigned int> run;
        mutable std::atomic<unsigned int> lumi;
        mutable std::atomic<unsigned int> strm;
        mutable std::atomic<unsigned int> work;
      };

      struct UnsafeCache {
        UnsafeCache() : value(0), run(0), lumi(0), strm(0), work(0) {}
        unsigned int value;
        unsigned int run;
        unsigned int lumi;
        unsigned int strm;
        unsigned int work;
      };

      struct Dummy {};

    }  //end anonymous namespace

    class StreamIntProducer : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>> {
    public:
      explicit StreamIntProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
      }

      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
        ++m_count;
        auto sCache = std::make_unique<UnsafeCache>();
        ++(sCache->strm);
        sCache->value = iID.value();
        return sCache;
      }

      void streamBeginRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        if (sCache->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
        if (sCache->run != 0 || sCache->lumi != 0 || sCache->work != 0 || sCache->strm != 1) {
          throw cms::Exception("out of sequence") << "streamBeginRun out of sequence in Stream " << iID.value();
        }
        ++(sCache->run);
      }

      void streamBeginLuminosityBlock(edm::StreamID iID,
                                      edm::LuminosityBlock const&,
                                      edm::EventSetup const&) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        if (sCache->lumi != 0 || sCache->work != 0) {
          throw cms::Exception("out of sequence")
              << "streamBeginLuminosityBlock out of sequence in Stream " << iID.value();
        }
        ++(sCache->lumi);
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        ++(sCache->work);
        if (sCache->lumi == 0 && sCache->run == 0) {
          throw cms::Exception("out of sequence") << "produce out of sequence in Stream " << iID.value();
        }
      }

      void streamEndLuminosityBlock(edm::StreamID iID,
                                    edm::LuminosityBlock const&,
                                    edm::EventSetup const&) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        --(sCache->lumi);
        sCache->work = 0;
        if (sCache->lumi != 0 || sCache->run == 0) {
          throw cms::Exception("out of sequence")
              << "streamEndLuminosityBlock out of sequence in Stream " << iID.value();
        }
      }

      void streamEndRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        --(sCache->run);
        sCache->work = 0;
        if (sCache->run != 0 || sCache->lumi != 0) {
          throw cms::Exception("out of sequence") << "streamEndRun out of sequence in Stream " << iID.value();
        }
      }

      void endStream(edm::StreamID iID) const override {
        ++m_count;
        auto sCache = streamCache(iID);
        --(sCache->strm);
        if (sCache->strm != 0 || sCache->run != 0 || sCache->lumi != 0) {
          throw cms::Exception("out of sequence") << "endStream out of sequence in Stream " << iID.value();
        }
      }

      ~StreamIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "StreamIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntProducer : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunCache<Cache>> {
    public:
      explicit RunIntProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunCache<Cache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
        produces<unsigned int>();
      }

      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        ++m_count;
        auto rCache = std::make_shared<Cache>();
        ++(rCache->run);
        return rCache;
      }

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override { return std::make_unique<UnsafeCache>(); }

      void streamBeginRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const override {
        auto rCache = runCache(iRun.index());
        if (rCache->run == 0) {
          throw cms::Exception("out of sequence") << "streamBeginRun before globalBeginRun in Stream " << iID.value();
        }
      }

      void produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const override {
        auto rCache = runCache(iEvent.getRun().index());
        ++(rCache->value);
      }

      void streamEndRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const override {
        auto rCache = runCache(iRun.index());
        if (rCache->run == 0) {
          throw cms::Exception("out of sequence") << "streamEndRun after globalEndRun in Stream " << iID.value();
        }
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        ++m_count;
        auto rCache = runCache(iRun.index());
        if (rCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "RunIntProducer cache value " << rCache->value << " but it was supposed to be " << cvalue_;
        }
        --(rCache->run);
      }

      ~RunIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntProducer
        : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockCache<Cache>> {
    public:
      explicit LumiIntProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockCache<Cache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
        produces<unsigned int>();
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                        edm::EventSetup const&) const override {
        ++m_count;
        auto lCache = std::make_shared<Cache>();
        ++(lCache->lumi);
        return lCache;
      }

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override { return std::make_unique<UnsafeCache>(); }

      void streamBeginLuminosityBlock(edm::StreamID iID,
                                      edm::LuminosityBlock const& iLB,
                                      edm::EventSetup const&) const override {
        auto lCache = luminosityBlockCache(iLB.index());
        if (lCache->lumi == 0) {
          throw cms::Exception("out of sequence")
              << "streamBeginLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
      }

      void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
        auto lCache = luminosityBlockCache(iEvent.getLuminosityBlock().index());
        ++(lCache->value);
      }

      void streamEndLuminosityBlock(edm::StreamID iID,
                                    edm::LuminosityBlock const& iLB,
                                    edm::EventSetup const&) const override {
        auto lCache = luminosityBlockCache(iLB.index());
        if (lCache->lumi == 0) {
          throw cms::Exception("out of sequence")
              << "streamEndLuminosityBlock seen before globalEndLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
        ++m_count;
        auto lCache = luminosityBlockCache(iLB.index());
        --(lCache->lumi);
        if (lCache->lumi != 0) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        if (lCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiIntProducer cache value " << lCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~LumiIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntProducer
        : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      explicit RunSummaryIntProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
        produces<unsigned int>();
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override { return std::make_unique<UnsafeCache>(); }

      std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const& iRun, edm::EventSetup const&) const override {
        ++m_count;
        auto gCache = std::make_shared<UnsafeCache>();
        ++(gCache->run);
        return gCache;
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        auto sCache = streamCache(iID);
        ++(sCache->value);
      }

      void streamEndRunSummary(edm::StreamID iID,
                               edm::Run const&,
                               edm::EventSetup const&,
                               UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->run == 0) {
          throw cms::Exception("out of sequence")
              << "streamEndRunSummary after globalEndRunSummary in Stream " << iID.value();
        }
        auto sCache = streamCache(iID);
        gCache->value += sCache->value;
        sCache->value = 0;
      }

      void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "RunSummaryIntProducer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
        --(gCache->run);
      }

      ~RunSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunSummaryIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntProducer
        : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      explicit LumiSummaryIntProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
        produces<unsigned int>();
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override { return std::make_unique<UnsafeCache>(); }

      std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                                                     edm::EventSetup const&) const override {
        ++m_count;
        auto gCache = std::make_shared<UnsafeCache>();
        gCache->lumi = iLB.luminosityBlockAuxiliary().luminosityBlock();
        return gCache;
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        auto sCache = streamCache(iID);
        ++(sCache->value);
      }

      void streamEndLuminosityBlockSummary(edm::StreamID iID,
                                           edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue")
              << "streamEndLuminosityBlockSummary unexpected lumi number in Stream " << iID.value();
        }
        auto sCache = streamCache(iID);
        gCache->value += sCache->value;
        sCache->value = 0;
      }

      void globalEndLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue") << "globalEndLuminosityBlockSummary unexpected lumi number";
        }
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiSummaryIntProducer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~LumiSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiSummaryIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryLumiProducer : public edm::limited::EDProducer<edm::StreamCache<UnsafeCache>,
                                                                    edm::LuminosityBlockSummaryCache<UnsafeCache>,
                                                                    edm::EndLuminosityBlockProducer> {
    public:
      explicit LumiSummaryLumiProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::StreamCache<UnsafeCache>,
                                     edm::LuminosityBlockSummaryCache<UnsafeCache>,
                                     edm::EndLuminosityBlockProducer>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
        produces<unsigned int>();
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override { return std::make_unique<UnsafeCache>(); }

      std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                                                     edm::EventSetup const&) const override {
        ++m_count;
        auto gCache = std::make_shared<UnsafeCache>();
        gCache->lumi = iLB.luminosityBlockAuxiliary().luminosityBlock();
        return gCache;
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        auto sCache = streamCache(iID);
        ++(sCache->value);
      }

      void streamEndLuminosityBlockSummary(edm::StreamID iID,
                                           edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue")
              << "streamEndLuminosityBlockSummary unexpected lumi number in Stream " << iID.value();
        }
        auto sCache = streamCache(iID);
        gCache->value += sCache->value;
        sCache->value = 0;
      }

      void globalEndLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue") << "globalEndLuminosityBlockSummary unexpected lumi number";
        }
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiSummaryLumiProducer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      void globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache const* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue") << "globalEndLuminosityBlockProduce unexpected lumi number";
        }
      }

      ~LumiSummaryLumiProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiSummaryLumiProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginRunProducer : public edm::limited::EDProducer<edm::RunCache<Dummy>, edm::BeginRunProducer> {
    public:
      explicit TestBeginRunProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::RunCache<Dummy>, edm::BeginRunProducer>(p),
            trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }

      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      mutable std::atomic<bool> brp{false};

      std::shared_ptr<Dummy> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        brp = false;
        return std::shared_ptr<Dummy>();
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        if (!brp) {
          throw cms::Exception("out of sequence") << "produce before globalBeginRunProduce in Stream " << iID.value();
        }
      }

      void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override {
        ++m_count;
        brp = true;
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {}

      ~TestBeginRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginRunProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndRunProducer : public edm::limited::EDProducer<edm::RunCache<Dummy>, edm::EndRunProducer> {
    public:
      explicit TestEndRunProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::RunCache<Dummy>, edm::EndRunProducer>(p),
            trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndRun>("a");
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      mutable std::atomic<bool> p{false};

      std::shared_ptr<Dummy> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        p = false;
        return std::shared_ptr<Dummy>();
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override { p = true; }

      void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override {
        if (!p) {
          throw cms::Exception("out of sequence") << "endRunProduce before produce";
        }
        ++m_count;
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {}

      ~TestEndRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndRunProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginLumiBlockProducer
        : public edm::limited::EDProducer<edm::LuminosityBlockCache<void>, edm::BeginLuminosityBlockProducer> {
    public:
      explicit TestBeginLumiBlockProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::LuminosityBlockCache<void>, edm::BeginLuminosityBlockProducer>(p),
            trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      mutable std::atomic<bool> gblp{false};

      std::shared_ptr<void> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                       edm::EventSetup const&) const override {
        gblp = false;
        return std::shared_ptr<void>();
      }

      void produce(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override {
        if (!gblp) {
          throw cms::Exception("out of sequence")
              << "produce before globalBeginLuminosityBlockProduce in Stream " << iID.value();
        }
      }

      void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
        ++m_count;
        gblp = true;
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {}

      ~TestBeginLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginLumiBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndLumiBlockProducer
        : public edm::limited::EDProducer<edm::LuminosityBlockCache<void>, edm::EndLuminosityBlockProducer> {
    public:
      explicit TestEndLumiBlockProducer(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::LuminosityBlockCache<void>, edm::EndLuminosityBlockProducer>(p),
            trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      mutable std::atomic<bool> p{false};

      std::shared_ptr<void> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                       edm::EventSetup const&) const override {
        p = false;
        return std::shared_ptr<void>();
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override { p = true; }

      void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
        if (!p) {
          throw cms::Exception("out of sequence") << "endLumiBlockProduce before produce";
        }
        ++m_count;
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {}

      ~TestEndLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndLumiBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestAccumulator : public edm::limited::EDProducer<edm::Accumulator> {
    public:
      explicit TestAccumulator(edm::ParameterSet const& p)
          : edm::limited::EDProducerBase(p),
            edm::limited::EDProducer<edm::Accumulator>(p),
            m_expectedCount(p.getParameter<unsigned int>("expectedCount")) {}

      void accumulate(edm::StreamID iID, edm::Event const&, edm::EventSetup const&) const override { ++m_count; }

      ~TestAccumulator() {
        if (m_count.load() != m_expectedCount) {
          throw cms::Exception("TestCount")
              << "TestAccumulator counter was " << m_count << " but it was supposed to be " << m_expectedCount;
        }
      }

      mutable std::atomic<unsigned int> m_count{0};
      const unsigned int m_expectedCount;
    };
  }  // namespace limited
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::limited::StreamIntProducer);
DEFINE_FWK_MODULE(edmtest::limited::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::limited::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::limited::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::limited::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::limited::LumiSummaryLumiProducer);
DEFINE_FWK_MODULE(edmtest::limited::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::limited::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::limited::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::limited::TestEndLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::limited::TestAccumulator);
