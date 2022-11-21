#ifndef FWCore_Framework_stream_EDAnalyzerAdaptor_h
#define FWCore_Framework_stream_EDAnalyzerAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzerAdaptor
//
/**\class edm::stream::EDAnalyzerAdaptor EDAnalyzerAdaptor.h "EDAnalyzerAdaptor.h"

 Description: Adapts an edm::stream::EDAnalyzer<> to work with an edm::Worker

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:18 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/callAbilities.h"
#include "FWCore/Framework/interface/stream/dummy_helpers.h"
#include "FWCore/Framework/interface/stream/makeGlobal.h"
#include "FWCore/Framework/interface/maker/MakeModuleHelper.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

// forward declarations

namespace edm {
  namespace stream {

    template <typename ABase, typename ModType>
    struct BaseToAdaptor;

    template <typename T>
    class EDAnalyzerAdaptor;
    template <typename ModType>
    struct BaseToAdaptor<EDAnalyzerAdaptorBase, ModType> {
      typedef EDAnalyzerAdaptor<ModType> Type;
    };

    template <typename T>
    class EDAnalyzerAdaptor : public EDAnalyzerAdaptorBase {
    public:
      EDAnalyzerAdaptor(edm::ParameterSet const& iPSet) : m_pset(&iPSet) {
        m_runs.resize(1);
        m_lumis.resize(1);
        m_runSummaries.resize(1);
        m_lumiSummaries.resize(1);
        typename T::GlobalCache const* dummy = nullptr;
        m_global = impl::makeGlobal<T>(iPSet, dummy);
        typename T::InputProcessBlockCache const* dummyInputProcessBlockCacheImpl = nullptr;
        m_inputProcessBlocks = impl::makeInputProcessBlockCacheImpl(dummyInputProcessBlockCacheImpl);
      }
      EDAnalyzerAdaptor(const EDAnalyzerAdaptor&) = delete;                   // stop default
      const EDAnalyzerAdaptor& operator=(const EDAnalyzerAdaptor&) = delete;  // stop default
      ~EDAnalyzerAdaptor() override { deleteModulesEarly(); }

      static void fillDescriptions(ConfigurationDescriptions& descriptions) { T::fillDescriptions(descriptions); }
      static void prevalidate(ConfigurationDescriptions& descriptions) { T::prevalidate(descriptions); }

      bool wantsProcessBlocks() const final { return T::HasAbility::kWatchProcessBlock; }
      bool wantsInputProcessBlocks() const final { return T::HasAbility::kInputProcessBlockCache; }
      bool wantsGlobalRuns() const final { return T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache; }
      bool wantsGlobalLuminosityBlocks() const final {
        return T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache;
      }

    private:
      using MyGlobal = CallGlobal<T>;
      using MyInputProcessBlock = CallInputProcessBlock<T>;
      using MyWatchProcessBlock = CallWatchProcessBlock<T>;
      using MyGlobalRun = CallGlobalRun<T>;
      using MyGlobalRunSummary = CallGlobalRunSummary<T>;
      using MyGlobalLuminosityBlock = CallGlobalLuminosityBlock<T>;
      using MyGlobalLuminosityBlockSummary = CallGlobalLuminosityBlockSummary<T>;

      void setupStreamModules() final {
        this->createStreamModules([this](unsigned int iStreamModule) -> EDAnalyzerBase* {
          auto tmp = impl::makeStreamModule<T>(*m_pset, m_global.get());
          MyGlobal::set(tmp, m_global.get());
          MyInputProcessBlock::set(tmp, &m_inputProcessBlocks, iStreamModule);
          return tmp;
        });
        m_pset = nullptr;
      }

      void preallocRuns(unsigned int iNRuns) final {
        m_runs.resize(iNRuns);
        m_runSummaries.resize(iNRuns);
      }
      void preallocLumis(unsigned int iNLumis) final {
        m_lumis.resize(iNLumis);
        m_lumiSummaries.resize(iNLumis);
      }

      void doBeginJob() final { MyGlobal::beginJob(m_global.get()); }
      void doEndJob() final { MyGlobal::endJob(m_global.get()); }
      void setupRun(EDAnalyzerBase* iProd, RunIndex iIndex) final { MyGlobalRun::set(iProd, m_runs[iIndex].get()); }
      void streamEndRunSummary(EDAnalyzerBase* iProd, edm::Run const& iRun, edm::EventSetup const& iES) final {
        auto s = m_runSummaries[iRun.index()].get();
        std::lock_guard<decltype(m_runSummaryLock)> guard(m_runSummaryLock);
        MyGlobalRunSummary::streamEndRunSummary(iProd, iRun, iES, s);
      }

      void setupLuminosityBlock(EDAnalyzerBase* iProd, LuminosityBlockIndex iIndex) final {
        MyGlobalLuminosityBlock::set(iProd, m_lumis[iIndex].get());
      }
      void streamEndLuminosityBlockSummary(EDAnalyzerBase* iProd,
                                           edm::LuminosityBlock const& iLumi,
                                           edm::EventSetup const& iES) final {
        auto s = m_lumiSummaries[iLumi.index()].get();
        std::lock_guard<decltype(m_lumiSummaryLock)> guard(m_lumiSummaryLock);
        MyGlobalLuminosityBlockSummary::streamEndLuminosityBlockSummary(iProd, iLumi, iES, s);
      }

      void doBeginProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kWatchProcessBlock) {
          ProcessBlock processBlock(pbp, moduleDescription(), mcc, false);
          processBlock.setConsumer(consumer());
          ProcessBlock const& cnstProcessBlock = processBlock;
          MyWatchProcessBlock::beginProcessBlock(cnstProcessBlock, m_global.get());
        }
      }

      void doAccessInputProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kInputProcessBlockCache) {
          ProcessBlock processBlock(pbp, moduleDescription(), mcc, false);
          processBlock.setConsumer(consumer());
          ProcessBlock const& cnstProcessBlock = processBlock;
          MyInputProcessBlock::accessInputProcessBlock(cnstProcessBlock, m_global.get(), m_inputProcessBlocks);
        }
      }

      void doEndProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kWatchProcessBlock) {
          ProcessBlock processBlock(pbp, moduleDescription(), mcc, true);
          processBlock.setConsumer(consumer());
          ProcessBlock const& cnstProcessBlock = processBlock;
          MyWatchProcessBlock::endProcessBlock(cnstProcessBlock, m_global.get());
        }
      }

      void doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache) {
          RunPrincipal const& rp = info.principal();
          Run r(rp, moduleDescription(), mcc, false);
          r.setConsumer(consumer());
          Run const& cnstR = r;
          RunIndex ri = rp.index();
          ESParentContext pc{mcc};
          const EventSetup c{info,
                             static_cast<unsigned int>(Transition::BeginRun),
                             this->consumer()->esGetTokenIndices(Transition::BeginRun),
                             pc};
          MyGlobalRun::beginRun(cnstR, c, m_global.get(), m_runs[ri]);
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          MyGlobalRunSummary::beginRun(cnstR, c, &rc, m_runSummaries[ri]);
        }
      }
      void doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache) {
          RunPrincipal const& rp = info.principal();
          Run r(rp, moduleDescription(), mcc, true);
          r.setConsumer(consumer());

          RunIndex ri = rp.index();
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          ESParentContext pc{mcc};
          const EventSetup c{info,
                             static_cast<unsigned int>(Transition::EndRun),
                             this->consumer()->esGetTokenIndices(Transition::EndRun),
                             pc};
          MyGlobalRunSummary::globalEndRun(r, c, &rc, m_runSummaries[ri].get());
          MyGlobalRun::endRun(r, c, &rc);
        }
      }

      void doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache) {
          LuminosityBlockPrincipal const& lbp = info.principal();
          LuminosityBlock lb(lbp, moduleDescription(), mcc, false);
          lb.setConsumer(consumer());
          LuminosityBlock const& cnstLb = lb;
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          ESParentContext pc{mcc};
          const EventSetup c{info,
                             static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                             this->consumer()->esGetTokenIndices(Transition::BeginLuminosityBlock),
                             pc};
          MyGlobalLuminosityBlock::beginLuminosityBlock(cnstLb, c, &rc, m_lumis[li]);
          typename T::LuminosityBlockContext lc(m_lumis[li].get(), m_runs[ri].get(), m_global.get());
          MyGlobalLuminosityBlockSummary::beginLuminosityBlock(cnstLb, c, &lc, m_lumiSummaries[li]);
        }
      }
      void doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache) {
          LuminosityBlockPrincipal const& lbp = info.principal();
          LuminosityBlock lb(lbp, moduleDescription(), mcc, true);
          lb.setConsumer(consumer());

          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::LuminosityBlockContext lc(m_lumis[li].get(), m_runs[ri].get(), m_global.get());
          ESParentContext pc{mcc};
          const EventSetup c{info,
                             static_cast<unsigned int>(Transition::EndLuminosityBlock),
                             this->consumer()->esGetTokenIndices(Transition::EndLuminosityBlock),
                             pc};
          MyGlobalLuminosityBlockSummary::globalEndLuminosityBlock(lb, c, &lc, m_lumiSummaries[li].get());
          MyGlobalLuminosityBlock::endLuminosityBlock(lb, c, &lc);
        }
      }

      void doRespondToCloseOutputFile() final { MyInputProcessBlock::clearCaches(m_inputProcessBlocks); }

      void selectInputProcessBlocks(ProductRegistry const& productRegistry,
                                    ProcessBlockHelperBase const& processBlockHelperBase) final {
        MyInputProcessBlock::selectInputProcessBlocks(
            m_inputProcessBlocks, productRegistry, processBlockHelperBase, *consumer());
      }

      // ---------- member data --------------------------------
      typename impl::choose_unique_ptr<typename T::GlobalCache>::type m_global;
      typename impl::choose_unique_ptr<typename T::InputProcessBlockCache>::type m_inputProcessBlocks;
      typename impl::choose_shared_vec<typename T::RunCache const>::type m_runs;
      typename impl::choose_shared_vec<typename T::LuminosityBlockCache const>::type m_lumis;
      typename impl::choose_shared_vec<typename T::RunSummaryCache>::type m_runSummaries;
      typename impl::choose_mutex<typename T::RunSummaryCache>::type m_runSummaryLock;
      typename impl::choose_shared_vec<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaries;
      typename impl::choose_mutex<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaryLock;
      ParameterSet const* m_pset;
    };
  }  // namespace stream

  template <>
  class MakeModuleHelper<edm::stream::EDAnalyzerAdaptorBase> {
    typedef edm::stream::EDAnalyzerAdaptorBase Base;

  public:
    template <typename ModType>
    static std::unique_ptr<Base> makeModule(ParameterSet const& pset) {
      typedef typename stream::BaseToAdaptor<Base, ModType>::Type Adaptor;
      auto module = std::make_unique<Adaptor>(pset);
      return std::unique_ptr<Base>(module.release());
    }
  };

}  // namespace edm

#endif
