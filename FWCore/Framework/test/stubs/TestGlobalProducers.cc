
/*----------------------------------------------------------------------

Toy edm::global::EDProducer modules of
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/

#include <atomic>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  namespace global {

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

    class StreamIntProducer : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>> {
    public:
      explicit StreamIntProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "global::StreamIntProducer " << desc.moduleLabel() << std::endl;
        });
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

    class RunIntProducer : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunCache<Cache>> {
    public:
      explicit RunIntProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
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
        : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockCache<Cache>> {
    public:
      explicit LumiIntProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
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
        : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      explicit RunSummaryIntProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
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
        : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      explicit LumiSummaryIntProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
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

    class LumiSummaryLumiProducer : public edm::global::EDProducer<edm::StreamCache<UnsafeCache>,
                                                                   edm::LuminosityBlockSummaryCache<UnsafeCache>,
                                                                   edm::EndLuminosityBlockProducer> {
    public:
      explicit LumiSummaryLumiProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
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

    class ProcessBlockIntProducer : public edm::global::EDProducer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int>();
        {
          auto tag = p.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = p.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntProducer::begin transitions " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;

        const unsigned int valueToGet = 11;
        if (not getTokenBegin_.isUninitialized()) {
          if (processBlock.get(getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
          }
        }
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << m_count;
        }
        ++m_count;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntProducer::end transitions " << m_count << " but it was supposed to be " << trans_;
        }
        {
          const unsigned int valueToGet = 11;
          if (not getTokenBegin_.isUninitialized()) {
            if (processBlock.get(getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 21;
          if (not getTokenEnd_.isUninitialized()) {
            if (processBlock.get(getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenEnd_);
            }
          }
        }
      }

      ~ProcessBlockIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDGetTokenT<unsigned int> getTokenBegin_;
      edm::EDGetTokenT<unsigned int> getTokenEnd_;
    };

    class TestBeginProcessBlockProducer : public edm::global::EDProducer<edm::BeginProcessBlockProducer> {
    public:
      explicit TestBeginProcessBlockProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")),
            token_(produces<unsigned int, edm::Transition::BeginProcessBlock>("begin")) {
        produces<unsigned int>();

        auto tag = p.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          getToken_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      void beginProcessBlockProduce(edm::ProcessBlock& processBlock) override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;
        const unsigned int valueToPutAndGet = 11;
        processBlock.emplace(token_, valueToPutAndGet);

        if (not getToken_.isUninitialized()) {
          if (processBlock.get(getToken_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(getToken_);
          }
        }
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlockProduce " << m_count;
        }
        ++m_count;
      }

      ~TestBeginProcessBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDPutTokenT<unsigned int> token_;
      edm::EDGetTokenT<unsigned int> getToken_;
    };

    class TestEndProcessBlockProducer : public edm::global::EDProducer<edm::EndProcessBlockProducer> {
    public:
      explicit TestEndProcessBlockProducer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")),
            token_(produces<unsigned int, edm::Transition::EndProcessBlock>("end")) {
        produces<unsigned int>();

        auto tag = p.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          getToken_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      void produce(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override { ++m_count; }

      void endProcessBlockProduce(edm::ProcessBlock& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndProcessBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
        const unsigned int valueToPutAndGet = 21;
        processBlock.emplace(token_, valueToPutAndGet);
        if (not getToken_.isUninitialized()) {
          if (processBlock.get(getToken_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(getToken_);
          }
        }
      }

      ~TestEndProcessBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "~TestEndProcessBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDPutTokenT<unsigned int> token_;
      edm::EDGetTokenT<unsigned int> getToken_;
    };

    class TestBeginRunProducer : public edm::global::EDProducer<edm::RunCache<Dummy>, edm::BeginRunProducer> {
    public:
      explicit TestBeginRunProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
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

    class TestEndRunProducer : public edm::global::EDProducer<edm::RunCache<Dummy>, edm::EndRunProducer> {
    public:
      explicit TestEndRunProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
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
        : public edm::global::EDProducer<edm::LuminosityBlockCache<void>, edm::BeginLuminosityBlockProducer> {
    public:
      explicit TestBeginLumiBlockProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
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
        : public edm::global::EDProducer<edm::LuminosityBlockCache<void>, edm::EndLuminosityBlockProducer> {
    public:
      explicit TestEndLumiBlockProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
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

    class TestAccumulator : public edm::global::EDProducer<edm::Accumulator, edm::EndLuminosityBlockProducer> {
    public:
      explicit TestAccumulator(edm::ParameterSet const& p)
          : m_expectedCount(p.getParameter<unsigned int>("expectedCount")),
            m_putToken(produces<edm::Transition::EndLuminosityBlock>()) {}

      void accumulate(edm::StreamID iID, edm::Event const&, edm::EventSetup const&) const override { ++m_count; }

      void globalEndLuminosityBlockProduce(edm::LuminosityBlock& l, edm::EventSetup const&) const override {
        l.emplace(m_putToken, m_count.load());
      }

      ~TestAccumulator() {
        if (m_count.load() != m_expectedCount) {
          throw cms::Exception("TestCount")
              << "TestAccumulator counter was " << m_count << " but it was supposed to be " << m_expectedCount;
        }
      }

      mutable std::atomic<unsigned int> m_count{0};
      const unsigned int m_expectedCount;
      const edm::EDPutTokenT<unsigned int> m_putToken;
    };

    class TestInputProcessBlockCache {
    public:
      long long int value_ = 0;
    };

    class TestInputProcessBlockCache1 {
    public:
      long long int value_ = 0;
    };

    class InputProcessBlockIntProducer
        : public edm::global::EDProducer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntProducer(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
        expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
        expectedSum_ = pset.getParameter<int>("expectedSum");
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlockM");
          if (not tag.label().empty()) {
            getTokenBeginM_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlockM");
          if (not tag.label().empty()) {
            getTokenEndM_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        registerProcessBlockCacheFiller<int>(
            getTokenBegin_, [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
              auto returnValue = std::make_shared<int>(0);
              *returnValue += processBlock.get(getTokenBegin_).value;
              *returnValue += processBlock.get(getTokenEnd_).value;
              ++transitions_;
              return returnValue;
            });
        registerProcessBlockCacheFiller<1>(getTokenBegin_,
                                           [this](edm::ProcessBlock const& processBlock,
                                                  std::shared_ptr<TestInputProcessBlockCache> const& previousCache) {
                                             auto returnValue = std::make_shared<TestInputProcessBlockCache>();
                                             returnValue->value_ += processBlock.get(getTokenBegin_).value;
                                             returnValue->value_ += processBlock.get(getTokenEnd_).value;
                                             ++transitions_;
                                             return returnValue;
                                           });
        registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
            getTokenBegin_,
            [this](edm::ProcessBlock const& processBlock,
                   std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              returnValue->value_ += processBlock.get(getTokenBegin_).value;
              returnValue->value_ += processBlock.get(getTokenEnd_).value;
              ++transitions_;
              return returnValue;
            });
      }

      void accessInputProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (processBlock.processName() == "PROD1") {
          sum_ += processBlock.get(getTokenBegin_).value;
          sum_ += processBlock.get(getTokenEnd_).value;
        }
        if (processBlock.processName() == "MERGE") {
          sum_ += processBlock.get(getTokenBeginM_).value;
          sum_ += processBlock.get(getTokenEndM_).value;
        }
        ++transitions_;
      }

      void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducer::produce cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducer::produce second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducer::produce third cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        ++transitions_;
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockIntProducer transitions " << transitions_
                                              << " but it was supposed to be " << expectedTransitions_;
        }
        if (sum_ != expectedSum_) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntProducer sum " << sum_ << " but it was supposed to be " << expectedSum_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntProducer cache size not zero at endJob " << cacheSize();
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      edm::EDGetTokenT<IntProduct> getTokenEndM_;
      mutable std::atomic<unsigned int> transitions_{0};
      int sum_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun_;
      int expectedSum_{0};
    };
  }  // namespace global
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::global::StreamIntProducer);
DEFINE_FWK_MODULE(edmtest::global::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryLumiProducer);
DEFINE_FWK_MODULE(edmtest::global::ProcessBlockIntProducer);
DEFINE_FWK_MODULE(edmtest::global::TestBeginProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::global::TestEndProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::global::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::global::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::global::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::global::TestEndLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::global::TestAccumulator);
DEFINE_FWK_MODULE(edmtest::global::InputProcessBlockIntProducer);
