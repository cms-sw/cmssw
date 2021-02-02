
/*----------------------------------------------------------------------

Toy edm::one::EDAnalyzer modules of
edm::one cache templates
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"

namespace edmtest {
  namespace one {

    class SharedResourcesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
    public:
      explicit SharedResourcesAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        usesResource();
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "one::SharedResourcesAnalyzer " << desc.moduleLabel() << std::endl;
        });
      }
      const unsigned int trans_;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override { ++m_count; }

      ~SharedResourcesAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "SharedResourcesAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchRunsAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
    public:
      explicit WatchRunsAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      bool br = false;
      bool er = false;
      const unsigned int trans_;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override {
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

      ~WatchRunsAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchLumiBlocksAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
    public:
      explicit WatchLumiBlocksAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        // just to create a data dependence
        auto const& tag = p.getParameter<edm::InputTag>("moduleLabel");
        if (not tag.label().empty()) {
          consumes<unsigned int, edm::InLumi>(tag);
        }
      }
      const unsigned int trans_;
      bool bl = false;
      bool el = false;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override {
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

      ~WatchLumiBlocksAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    namespace an {
      struct Cache {
        bool begin = true;
        bool end = false;
      };
    }  // namespace an
    class RunCacheAnalyzer : public edm::one::EDAnalyzer<edm::RunCache<an::Cache>> {
    public:
      explicit RunCacheAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {
        ++m_count;
        auto c = runCache(iEvent.getRun().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
      }

      std::shared_ptr<an::Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<an::Cache>();
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

      ~RunCacheAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiBlockCacheAnalyzer : public edm::one::EDAnalyzer<edm::LuminosityBlockCache<an::Cache>> {
    public:
      explicit LumiBlockCacheAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {
        ++m_count;

        auto c = luminosityBlockCache(iEvent.getLuminosityBlock().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
      }

      std::shared_ptr<an::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                            edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<an::Cache>();
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

      ~LumiBlockCacheAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer : public edm::one::EDAnalyzer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& pset) : trans_(pset.getParameter<int>("transitions")) {
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::begin transitions " << m_count << " but it was supposed to be " << 0;
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

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "analyze before beginProcessBlock " << m_count;
        }
        ++m_count;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::end transitions " << m_count << " but it was supposed to be " << trans_;
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

  }  // namespace one
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::RunCacheAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::LumiBlockCacheAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::ProcessBlockIntAnalyzer);
