
/*----------------------------------------------------------------------

Toy edm::one::EDProducer modules of 
edm::one cache classes and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {
  namespace one {

    class SharedResourcesProducer : public edm::one::EDProducer<edm::one::SharedResources> {
    public:
      explicit SharedResourcesProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        usesResource();
      }
      const unsigned int trans_;
      unsigned int m_count = 0;

      void produce(edm::Event&, edm::EventSetup const&) override { ++m_count; }

      ~SharedResourcesProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "SharedResourcesProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchRunsProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
    public:
      explicit WatchRunsProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
      }
      bool br = false;
      bool er = false;
      const unsigned int trans_;
      unsigned int m_count = 0;

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (!br || er) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        ++m_count;
        if (br) {
          throw cms::Exception("out of sequence") << " beginRun seen multiple times";
        }
        br = true;
        er = false;
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {
        ++m_count;
        if (!br) {
          throw cms::Exception("out of sequence") << " endRun before beginRun";
        }
        br = false;
        er = true;
      }

      ~WatchRunsProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchLumiBlocksProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks> {
    public:
      explicit WatchLumiBlocksProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
      }
      const unsigned int trans_;
      bool bl = false;
      bool el = false;
      unsigned int m_count = 0;

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (!bl || el) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        ++m_count;
        if (bl) {
          throw cms::Exception("out of sequence") << " beginLumiBlock seen mutiple times";
        }
        bl = true;
        el = false;
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        ++m_count;
        if (!bl) {
          throw cms::Exception("out of sequence") << " endLumiBlock before beginLumiBlock";
        }
        bl = false;
        el = true;
      }

      ~WatchLumiBlocksProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    namespace prdr {
      struct Cache {
        bool begin = true;
        bool end = false;
      };
    }  // namespace prdr
    class RunCacheProducer : public edm::one::EDProducer<edm::RunCache<prdr::Cache>> {
    public:
      explicit RunCacheProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void produce(edm::Event& iEvent, edm::EventSetup const&) override {
        ++m_count;
        auto c = runCache(iEvent.getRun().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
      }

      std::shared_ptr<prdr::Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<prdr::Cache>();
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) final {
        ++m_count;
        auto c = runCache(iRun.index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in globalEndRun";
        }
        if (!c->begin) {
          throw cms::Exception("out of sequence") << " endRun before beginRun";
        }
        c->begin = false;
        c->end = true;
      }

      ~RunCacheProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiBlockCacheProducer : public edm::one::EDProducer<edm::LuminosityBlockCache<prdr::Cache>> {
    public:
      explicit LumiBlockCacheProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void produce(edm::Event& iEvent, edm::EventSetup const&) override {
        ++m_count;

        auto c = luminosityBlockCache(iEvent.getLuminosityBlock().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
      }

      std::shared_ptr<prdr::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                              edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<prdr::Cache>();
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) override {
        ++m_count;
        auto c = luminosityBlockCache(iLumi.index());
        if (!c->begin) {
          throw cms::Exception("out of sequence") << " endLumiBlock before beginLumiBlock";
        }
        c->begin = false;
        c->end = true;
      }

      ~LumiBlockCacheProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginRunProducer : public edm::one::EDProducer<edm::one::WatchRuns, edm::BeginRunProducer> {
    public:
      explicit TestBeginRunProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }
      const unsigned int trans_;
      unsigned int m_count = 0;
      bool p = false;

      void beginRun(edm::Run const&, edm::EventSetup const&) override { p = false; }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        p = true;
      }

      void beginRunProduce(edm::Run&, edm::EventSetup const&) override {
        if (p) {
          throw cms::Exception("out of sequence") << "produce before beginRunProduce";
        }
        ++m_count;
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {}

      ~TestBeginRunProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginRunProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginLumiBlockProducer
        : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks, edm::BeginLuminosityBlockProducer> {
    public:
      explicit TestBeginLumiBlockProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }
      const unsigned int trans_;
      unsigned int m_count = 0;
      bool p = false;

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override { p = false; }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        p = true;
      }

      void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
        ++m_count;
        if (p) {
          throw cms::Exception("out of sequence") << "produce before beginLuminosityBlockProduce";
        }
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}

      ~TestBeginLumiBlockProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginLumiBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndRunProducer : public edm::one::EDProducer<edm::one::WatchRuns, edm::EndRunProducer> {
    public:
      explicit TestEndRunProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        produces<unsigned int, edm::Transition::EndRun>("a");
      }
      const unsigned int trans_;
      bool erp = false;

      void beginRun(edm::Run const&, edm::EventSetup const&) override { erp = false; }

      unsigned int m_count = 0;

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (erp) {
          throw cms::Exception("out of sequence") << "endRunProduce before produce";
        }
      }

      void endRunProduce(edm::Run&, edm::EventSetup const&) override {
        ++m_count;
        erp = true;
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {}

      ~TestEndRunProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndRunProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndLumiBlockProducer
        : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks, edm::EndLuminosityBlockProducer> {
    public:
      explicit TestEndLumiBlockProducer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }
      const unsigned int trans_;
      bool elbp = false;
      unsigned int m_count = 0;

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override { elbp = false; }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (elbp) {
          throw cms::Exception("out of sequence") << "endLumiBlockProduce before produce";
        }
      }

      void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
        ++m_count;
        elbp = true;
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}

      ~TestEndLumiBlockProducer() noexcept(false) {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndLumiBlockProducer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestAccumulator : public edm::one::EDProducer<edm::Accumulator> {
    public:
      explicit TestAccumulator(edm::ParameterSet const& p)
          : m_expectedCount(p.getParameter<unsigned int>("expectedCount")) {}

      void accumulate(edm::Event const&, edm::EventSetup const&) override { ++m_count; }

      ~TestAccumulator() {
        if (m_count.load() != m_expectedCount) {
          throw cms::Exception("TestCount")
              << "TestAccumulator counter was " << m_count << " but it was supposed to be " << m_expectedCount;
        }
      }

      mutable std::atomic<unsigned int> m_count{0};
      const unsigned int m_expectedCount;
    };

  }  // namespace one
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesProducer);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsProducer);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksProducer);
DEFINE_FWK_MODULE(edmtest::one::RunCacheProducer);
DEFINE_FWK_MODULE(edmtest::one::LumiBlockCacheProducer);
DEFINE_FWK_MODULE(edmtest::one::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::one::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::one::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::one::TestEndLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::one::TestAccumulator);
