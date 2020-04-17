
/*----------------------------------------------------------------------

Toy edm::global::EDAnalyzer modules of
edm::*Cache templates
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"

namespace edmtest {
  namespace global {

    namespace {
      struct Cache {
        Cache() : value(0) {}
        //Using mutable since we want to update the value.
        mutable std::atomic<unsigned int> value;
      };

      struct UnsafeCache {
        UnsafeCache() : value(0), lumi(0) {}
        unsigned int value;
        unsigned int lumi;
      };

    }  //end anonymous namespace

    class StreamIntAnalyzer : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>> {
    public:
      explicit StreamIntAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "global::StreamIntAnalyzer " << desc.moduleLabel() << std::endl;
        });
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
        ++m_count;
        auto pCache = std::make_unique<UnsafeCache>();
        pCache->value = iID.value();
        return pCache;
      }

      void streamBeginRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamBeginLuminosityBlock(edm::StreamID iID,
                                      edm::LuminosityBlock const&,
                                      edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamEndLuminosityBlock(edm::StreamID iID,
                                    edm::LuminosityBlock const&,
                                    edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamEndRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void endStream(edm::StreamID iID) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      ~StreamIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "StreamIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntAnalyzer : public edm::global::EDAnalyzer<edm::RunCache<Cache>> {
    public:
      explicit RunIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<Cache>();
      }

      void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++((runCache(iEvent.getRun().index()))->value);
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        ++m_count;
        if ((runCache(iRun.index()))->value != cvalue_) {
          throw cms::Exception("cache value") << "RunIntAnalyzer cache value " << (runCache(iRun.index()))->value
                                              << " but it was supposed to be " << cvalue_;
        }
      }

      ~RunIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntAnalyzer : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
    public:
      explicit LumiIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
        // just to create a data dependence
        auto const& tag = p.getParameter<edm::InputTag>("moduleLabel");
        if (not tag.label().empty()) {
          consumes<unsigned int, edm::InLumi>(tag);
        }
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                        edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<Cache>();
      }

      void analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++(luminosityBlockCache(iEvent.getLuminosityBlock().index())->value);
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
        ++m_count;
        if ((luminosityBlockCache(iLB.index()))->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiIntAnalyzer cache value " << (luminosityBlockCache(iLB.index()))->value
              << " but it was supposed to be " << cvalue_;
        }
      }

      ~LumiIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntAnalyzer
        : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      explicit RunSummaryIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<UnsafeCache>();
      }

      void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
        ++m_count;
        ++((streamCache(iID))->value);
      }

      void streamEndRunSummary(edm::StreamID iID,
                               edm::Run const&,
                               edm::EventSetup const&,
                               UnsafeCache* gCache) const override {
        ++m_count;
        gCache->value += (streamCache(iID))->value;
        (streamCache(iID))->value = 0;
      }

      void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "RunSummaryIntAnalyzer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~RunSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntAnalyzer
        : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      explicit LumiSummaryIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                                                     edm::EventSetup const&) const override {
        ++m_count;
        auto gCache = std::make_shared<UnsafeCache>();
        gCache->lumi = iLB.luminosityBlockAuxiliary().luminosityBlock();
        return gCache;
      }

      void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++((streamCache(iID))->value);
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
        gCache->value += (streamCache(iID))->value;
        (streamCache(iID))->value = 0;
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
              << "LumiSummaryIntAnalyzer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~LumiSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer : public edm::global::EDAnalyzer<edm::WatchProcessBlock,
                                                                   edm::StreamCache<UnsafeCache>,
                                                                   edm::RunCache<UnsafeCache>> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
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

      void beginJob() override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::beginJob transition " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;
      }

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        if (m_count < 1) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::beginStream transition " << m_count
                                              << " but it was supposed to be at least 1";
        }
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) const override {
        if (m_count != 5) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::beginProcessBlock transition " << m_count
                                              << " but it was supposed to be " << 5;
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

      std::shared_ptr<UnsafeCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
        if (m_count < 6u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::globalBeginRun transition " << m_count
                                              << " but it was supposed to be at least 6";
        }
        ++m_count;
        return std::make_shared<UnsafeCache>();
      }

      void analyze(edm::StreamID iID, edm::Event const&, edm::EventSetup const&) const override {
        if (m_count < 7u) {
          throw cms::Exception("out of sequence") << "analyze before beginProcessBlock " << m_count;
        }
        ++m_count;
      }

      void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
        if (m_count < 15u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::globalEndRun transition " << m_count
                                              << " but it was supposed to be at least 15";
        }
        ++m_count;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) const override {
        if (m_count != 646u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::endProcessBlock transition " << m_count
                                              << " but it was supposed to be " << 646;
        }
        ++m_count;
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

      void endStream(edm::StreamID) const override {
        if (m_count < 647u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::endStream transition " << m_count
                                              << " but it was supposed to be at least 647";
        }
        ++m_count;
      }

      void endJob() override {
        if (m_count != 651u) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::endJob transition " << m_count << " but it was supposed to be " << 651;
        }
        ++m_count;
      }

      ~ProcessBlockIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDGetTokenT<unsigned int> getTokenBegin_;
      edm::EDGetTokenT<unsigned int> getTokenEnd_;
    };

  }  // namespace global
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::global::StreamIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::ProcessBlockIntAnalyzer);
