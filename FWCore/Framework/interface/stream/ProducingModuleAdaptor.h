#ifndef FWCore_Framework_stream_ProducingModuleAdaptor_h
#define FWCore_Framework_stream_ProducingModuleAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ProducingModuleAdaptor
//
/**\class edm::stream::ProducingModuleAdaptor ProducingModuleAdaptor.h "ProducingModuleAdaptor.h"

 Description: Adapts an edm::stream::EDProducer<> to work with an edm::Worker

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:18 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/callAbilities.h"
#include "FWCore/Framework/interface/stream/dummy_helpers.h"
#include "FWCore/Framework/interface/stream/makeGlobal.h"
// forward declarations

namespace edm {
  class ConfigurationDescriptions;
  namespace stream {

    template <typename T, typename M, typename B>
    class ProducingModuleAdaptor : public B {
    public:
      ProducingModuleAdaptor(edm::ParameterSet const& iPSet) : m_pset(&iPSet) {
        m_runs.resize(1);
        m_lumis.resize(1);
        m_runSummaries.resize(1);
        m_lumiSummaries.resize(1);
        typename T::GlobalCache const* dummy = nullptr;
        m_global = impl::makeGlobal<T>(iPSet, dummy);
      }
      ~ProducingModuleAdaptor() override {}

      static void fillDescriptions(ConfigurationDescriptions& descriptions) { T::fillDescriptions(descriptions); }
      static void prevalidate(ConfigurationDescriptions& descriptions) { T::prevalidate(descriptions); }

      bool wantsGlobalRuns() const final {
        return T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache or T::HasAbility::kBeginRunProducer or
               T::HasAbility::kEndRunProducer;
      }
      bool wantsGlobalLuminosityBlocks() const final {
        return T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache or
               T::HasAbility::kBeginLuminosityBlockProducer or T::HasAbility::kEndLuminosityBlockProducer;
      }

      bool hasAcquire() const final { return T::HasAbility::kExternalWork; }

      bool hasAccumulator() const final { return T::HasAbility::kAccumulator; }

    private:
      typedef CallGlobal<T> MyGlobal;
      typedef CallGlobalRun<T> MyGlobalRun;
      typedef CallGlobalRunSummary<T> MyGlobalRunSummary;
      typedef CallBeginRunProduce<T> MyBeginRunProduce;
      typedef CallEndRunProduce<T> MyEndRunProduce;
      typedef CallGlobalLuminosityBlock<T> MyGlobalLuminosityBlock;
      typedef CallGlobalLuminosityBlockSummary<T> MyGlobalLuminosityBlockSummary;
      typedef CallBeginLuminosityBlockProduce<T> MyBeginLuminosityBlockProduce;
      typedef CallEndLuminosityBlockProduce<T> MyEndLuminosityBlockProduce;

      void setupStreamModules() final {
        this->createStreamModules([this]() -> M* {
          auto tmp = impl::makeStreamModule<T>(*m_pset, m_global.get());
          MyGlobal::set(tmp, m_global.get());
          return tmp;
        });
        m_pset = nullptr;
      }

      void preallocLumis(unsigned int iNLumis) final {
        m_lumis.resize(iNLumis);
        m_lumiSummaries.resize(iNLumis);
      }
      void doEndJob() final { MyGlobal::endJob(m_global.get()); }
      void setupRun(M* iProd, RunIndex iIndex) final { MyGlobalRun::set(iProd, m_runs[iIndex].get()); }
      void streamEndRunSummary(M* iProd, edm::Run const& iRun, edm::EventSetup const& iES) final {
        auto s = m_runSummaries[iRun.index()].get();
        std::lock_guard<decltype(m_runSummaryLock)> guard(m_runSummaryLock);
        MyGlobalRunSummary::streamEndRunSummary(iProd, iRun, iES, s);
      }

      void setupLuminosityBlock(M* iProd, LuminosityBlockIndex iIndex) final {
        MyGlobalLuminosityBlock::set(iProd, m_lumis[iIndex].get());
      }
      void streamEndLuminosityBlockSummary(M* iProd,
                                           edm::LuminosityBlock const& iLumi,
                                           edm::EventSetup const& iES) final {
        auto s = m_lumiSummaries[iLumi.index()].get();
        std::lock_guard<decltype(m_lumiSummaryLock)> guard(m_lumiSummaryLock);
        MyGlobalLuminosityBlockSummary::streamEndLuminosityBlockSummary(iProd, iLumi, iES, s);
      }

      void doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache or T::HasAbility::kBeginRunProducer) {
          Run r(rp, this->moduleDescription(), mcc, false);
          r.setConsumer(this->consumer());
          r.setProducer(this->producer());
          Run const& cnstR = r;
          RunIndex ri = rp.index();
          const EventSetup c{ci,
                             static_cast<unsigned int>(Transition::BeginRun),
                             this->consumer()->esGetTokenIndices(Transition::BeginRun),
                             false};
          MyGlobalRun::beginRun(cnstR, c, m_global.get(), m_runs[ri]);
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          MyGlobalRunSummary::beginRun(cnstR, c, &rc, m_runSummaries[ri]);
          if constexpr (T::HasAbility::kBeginRunProducer) {
            MyBeginRunProduce::produce(r, c, &rc);
            this->commit(r);
          }
        }
      }
      void doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache or T::HasAbility::kEndRunProducer) {
          Run r(rp, this->moduleDescription(), mcc, true);
          r.setConsumer(this->consumer());
          r.setProducer(this->producer());

          RunIndex ri = rp.index();
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          const EventSetup c{ci,
                             static_cast<unsigned int>(Transition::EndRun),
                             this->consumer()->esGetTokenIndices(Transition::EndRun),
                             false};
          MyGlobalRunSummary::globalEndRun(r, c, &rc, m_runSummaries[ri].get());
          if constexpr (T::HasAbility::kEndRunProducer) {
            MyEndRunProduce::produce(r, c, &rc, m_runSummaries[ri].get());
            this->commit(r);
          }
          MyGlobalRun::endRun(r, c, &rc);
        }
      }

      void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                  EventSetupImpl const& ci,
                                  ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache or
                      T::HasAbility::kBeginLuminosityBlockProducer) {
          LuminosityBlock lb(lbp, this->moduleDescription(), mcc, false);
          lb.setConsumer(this->consumer());
          lb.setProducer(this->producer());
          LuminosityBlock const& cnstLb = lb;
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::RunContext rc(m_runs[ri].get(), m_global.get());
          const EventSetup c{ci,
                             static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                             this->consumer()->esGetTokenIndices(Transition::BeginLuminosityBlock),
                             false};

          MyGlobalLuminosityBlock::beginLuminosityBlock(cnstLb, c, &rc, m_lumis[li]);
          typename T::LuminosityBlockContext lc(m_lumis[li].get(), m_runs[ri].get(), m_global.get());
          MyGlobalLuminosityBlockSummary::beginLuminosityBlock(cnstLb, c, &lc, m_lumiSummaries[li]);
          if constexpr (T::HasAbility::kBeginLuminosityBlockProducer) {
            MyBeginLuminosityBlockProduce::produce(lb, c, &lc);
            this->commit(lb);
          }
        }
      }
      void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                EventSetupImpl const& ci,
                                ModuleCallingContext const* mcc) final {
        if constexpr (T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache or
                      T::HasAbility::kEndLuminosityBlockProducer) {
          LuminosityBlock lb(lbp, this->moduleDescription(), mcc, true);
          lb.setConsumer(this->consumer());
          lb.setProducer(this->producer());

          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::LuminosityBlockContext lc(m_lumis[li].get(), m_runs[ri].get(), m_global.get());
          const EventSetup c{ci,
                             static_cast<unsigned int>(Transition::EndLuminosityBlock),
                             this->consumer()->esGetTokenIndices(Transition::EndLuminosityBlock),
                             false};
          MyGlobalLuminosityBlockSummary::globalEndLuminosityBlock(lb, c, &lc, m_lumiSummaries[li].get());
          if constexpr (T::HasAbility::kEndLuminosityBlockProducer) {
            MyEndLuminosityBlockProduce::produce(lb, c, &lc, m_lumiSummaries[li].get());
            this->commit(lb);
          }
          MyGlobalLuminosityBlock::endLuminosityBlock(lb, c, &lc);
        }
      }

      ProducingModuleAdaptor(const ProducingModuleAdaptor&) = delete;  // stop default

      const ProducingModuleAdaptor& operator=(const ProducingModuleAdaptor&) = delete;  // stop default

      // ---------- member data --------------------------------
      typename impl::choose_unique_ptr<typename T::GlobalCache>::type m_global;
      typename impl::choose_shared_vec<typename T::RunCache const>::type m_runs;
      typename impl::choose_shared_vec<typename T::LuminosityBlockCache const>::type m_lumis;
      typename impl::choose_shared_vec<typename T::RunSummaryCache>::type m_runSummaries;
      typename impl::choose_mutex<typename T::RunSummaryCache>::type m_runSummaryLock;
      typename impl::choose_shared_vec<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaries;
      typename impl::choose_mutex<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaryLock;
      ParameterSet const* m_pset;
    };
  }  // namespace stream
}  // namespace edm

#endif
