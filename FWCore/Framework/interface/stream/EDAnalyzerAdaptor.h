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
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/callAbilities.h"
#include "FWCore/Framework/interface/stream/dummy_helpers.h"
#include "FWCore/Framework/src/MakeModuleHelper.h"

// forward declarations

namespace edm {
  namespace stream {

    namespace impl {
      template<typename T, typename G>
      std::unique_ptr<G> makeGlobal(edm::ParameterSet const& iPSet, G const*) {
        return T::initializeGlobalCache(iPSet);
      }
      template<typename T>
      dummy_ptr makeGlobal(edm::ParameterSet const& iPSet, void const*) {
        return dummy_ptr();
      }
      
      template< typename T, typename G>
      T* makeStreamModule(edm::ParameterSet const& iPSet,
                                          G const* iGlobal) {
        return new T(iPSet,iGlobal);
      }

      template< typename T>
      T* makeStreamModule(edm::ParameterSet const& iPSet,
                                          void const* ) {
        return new T(iPSet);
      }
    }

    template<typename ABase, typename ModType> struct BaseToAdaptor;

    template<typename T> class EDAnalyzerAdaptor;
    template<typename ModType> struct BaseToAdaptor<EDAnalyzerAdaptorBase,ModType> {
      typedef EDAnalyzerAdaptor<ModType> Type;
    };

    template<typename T>
    class EDAnalyzerAdaptor : public EDAnalyzerAdaptorBase
    {
      
    public:
      EDAnalyzerAdaptor( edm::ParameterSet const& iPSet):
      m_pset(&iPSet)
      {
        m_runs.resize(1);
        m_lumis.resize(1);
        m_runSummaries.resize(1);
        m_lumiSummaries.resize(1);
        typename T::GlobalCache const* dummy=nullptr;
        m_global.reset( impl::makeGlobal<T>(iPSet,dummy).release());
      }
      ~EDAnalyzerAdaptor() {
      }
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions) {
        T::fillDescriptions(descriptions);
      }
      static void prevalidate(ConfigurationDescriptions& descriptions) {
        T::prevalidate(descriptions);
      }

      
    private:
      typedef CallGlobal<T> MyGlobal;
      typedef CallGlobalRun<T> MyGlobalRun;
      typedef CallGlobalRunSummary<T> MyGlobalRunSummary;
      typedef CallGlobalLuminosityBlock<T> MyGlobalLuminosityBlock;
      typedef CallGlobalLuminosityBlockSummary<T> MyGlobalLuminosityBlockSummary;
      
      void setupStreamModules() override final {
        this->createStreamModules([this] () -> EDAnalyzerBase* {
          auto tmp = impl::makeStreamModule<T>(*m_pset,m_global.get());
          MyGlobal::set(tmp,m_global.get());
          return tmp;
        });
        m_pset= nullptr;
      }

      void doEndJob() override final {
        MyGlobal::endJob(m_global.get());
      }
      void setupRun(EDAnalyzerBase* iProd, RunIndex iIndex) override final {
        MyGlobalRun::set(iProd, m_runs[iIndex].get());
      }
      void streamEndRunSummary(EDAnalyzerBase* iProd,
                               edm::Run const& iRun,
                               edm::EventSetup const& iES) override final {
        auto s = m_runSummaries[iRun.index()].get();
        MyGlobalRunSummary::streamEndRunSummary(iProd,iRun,iES,s);
      }
 
      void setupLuminosityBlock(EDAnalyzerBase* iProd, LuminosityBlockIndex iIndex) override final
      {
        MyGlobalLuminosityBlock::set(iProd, m_lumis[iIndex].get());
      }
      void streamEndLuminosityBlockSummary(EDAnalyzerBase* iProd,
                                           edm::LuminosityBlock const& iLumi,
                                           edm::EventSetup const& iES) override final {
        auto s = m_lumiSummaries[iLumi.index()].get();
        MyGlobalLuminosityBlockSummary::streamEndLuminosityBlockSummary(iProd,iLumi,iES,s);
      }

      void doBeginRun(RunPrincipal& rp,
                      EventSetup const& c,
                      ModuleCallingContext const* mcc) override final {
        if(T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache) {
          Run r(rp, moduleDescription(), mcc);
          r.setConsumer(consumer());
          Run const& cnstR = r;
          RunIndex ri = rp.index();
          MyGlobalRun::beginRun(cnstR,c,m_global.get(),m_runs[ri]);
          typename T::RunContext rc(m_runs[ri].get(),m_global.get());
          MyGlobalRunSummary::beginRun(cnstR,c,&rc,m_runSummaries[ri]);
        }
      }
      void doEndRun(RunPrincipal& rp,
                    EventSetup const& c,
                    ModuleCallingContext const* mcc) override final
      {
        if(T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache) {
          
          Run r(rp, moduleDescription(), mcc);
          r.setConsumer(consumer());

          RunIndex ri = rp.index();
          typename T::RunContext rc(m_runs[ri].get(),m_global.get());
          MyGlobalRunSummary::globalEndRun(r,c,&rc,m_runSummaries[ri].get());
          MyGlobalRun::endRun(r,c,&rc);
        }
      }

      void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp,
                                  EventSetup const& c,
                                  ModuleCallingContext const* mcc) override final
      {
        if(T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache) {
          LuminosityBlock lb(lbp, moduleDescription(), mcc);
          lb.setConsumer(consumer());
          LuminosityBlock const& cnstLb = lb;
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::RunContext rc(m_runs[ri].get(),m_global.get());
          MyGlobalLuminosityBlock::beginLuminosityBlock(cnstLb,c,&rc,m_lumis[li]);
          typename T::LuminosityBlockContext lc(m_lumis[li].get(),m_runs[ri].get(),m_global.get());
          MyGlobalLuminosityBlockSummary::beginLuminosityBlock(cnstLb,c,&lc,m_lumiSummaries[li]);
        }
        
      }
      void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp,
                                EventSetup const& c,
                                ModuleCallingContext const* mcc) override final {
        if(T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache) {
          
          LuminosityBlock lb(lbp, moduleDescription(), mcc);
          lb.setConsumer(consumer());
          
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::LuminosityBlockContext lc(m_lumis[li].get(),m_runs[ri].get(),m_global.get());
          MyGlobalLuminosityBlockSummary::globalEndLuminosityBlock(lb,c,&lc,m_lumiSummaries[li].get());
          MyGlobalLuminosityBlock::endLuminosityBlock(lb,c,&lc);
        }
      }

      EDAnalyzerAdaptor(const EDAnalyzerAdaptor&); // stop default
      
      const EDAnalyzerAdaptor& operator=(const EDAnalyzerAdaptor&); // stop default
      
      // ---------- member data --------------------------------
      typename impl::choose_unique_ptr<typename T::GlobalCache>::type m_global;
      typename impl::choose_shared_vec<typename T::RunCache const>::type m_runs;
      typename impl::choose_shared_vec<typename T::LuminosityBlockCache const>::type m_lumis;
      typename impl::choose_shared_vec<typename T::RunSummaryCache>::type m_runSummaries;
      typename impl::choose_shared_vec<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaries;
      ParameterSet const* m_pset;
    };
  }
  
  template<>
  class MakeModuleHelper<edm::stream::EDAnalyzerAdaptorBase>
  {
    typedef edm::stream::EDAnalyzerAdaptorBase Base;
  public:
    template<typename ModType>
    static std::unique_ptr<Base> makeModule(ParameterSet const& pset) {
      typedef typename stream::BaseToAdaptor<Base,ModType>::Type Adaptor;
      std::unique_ptr<Adaptor> module = std::unique_ptr<Adaptor>(new Adaptor(pset));
      return std::unique_ptr<Base>(module.release());
    }
  };

}

#endif
