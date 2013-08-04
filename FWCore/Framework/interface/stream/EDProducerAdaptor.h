#ifndef FWCore_Framework_stream_EDProducerAdaptor_h
#define FWCore_Framework_stream_EDProducerAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDProducerAdaptor
// 
/**\class edm::stream::EDProducerAdaptor EDProducerAdaptor.h "EDProducerAdaptor.h"

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
#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/callAbilities.h"
#include "FWCore/Framework/interface/stream/dummy_helpers.h"
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
    
    template<typename T>
    class EDProducerAdaptor : public EDProducerAdaptorBase
    {
      
    public:
      EDProducerAdaptor( edm::ParameterSet const& iPSet)
      {
        m_runs.resize(1);
        m_lumis.resize(1);
        typename T::GlobalCache const* dummy=nullptr;
        m_global.reset( impl::makeGlobal<T>(iPSet,dummy).release());
        this->createStreamModules([this,&iPSet] () -> EDProducerBase* {return impl::makeStreamModule<T>(iPSet,m_global.get());});
      }
      ~EDProducerAdaptor() {
      }
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions) {
        T::fillDescriptions(descriptions);
      }
      static void prevalidate(ConfigurationDescriptions& descriptions) {
        T::prevalidate(descriptions);
      }

      
    private:
      typedef CallGlobal<T,T::HasAbility::kGlobalCache> MyGlobal;
      typedef CallGlobalRun<T,T::HasAbility::kRunCache> MyGlobalRun;
      typedef CallGlobalRunSummary<T,T::HasAbility::kRunSummaryCache> MyGlobalRunSummary;
      typedef CallBeginRunProduce<T,T::HasAbility::kBeginRunProducer> MyBeginRunProduce;
      typedef CallEndRunProduce<T,T::HasAbility::kEndRunProducer, T::HasAbility::kRunSummaryCache> MyEndRunProduce;
      typedef CallGlobalLuminosityBlock<T,T::HasAbility::kLuminosityBlockCache> MyGlobalLuminosityBlock;
      typedef CallGlobalLuminosityBlockSummary<T,T::HasAbility::kLuminosityBlockSummaryCache> MyGlobalLuminosityBlockSummary;
      typedef CallBeginLuminosityBlockProduce<T,T::HasAbility::kBeginLuminosityBlockProducer> MyBeginLuminosityBlockProduce;
      typedef CallEndLuminosityBlockProduce<T,T::HasAbility::kEndLuminosityBlockProducer, T::HasAbility::kLuminosityBlockSummaryCache> MyEndLuminosityBlockProduce;
      
      void doEndJob() override final {
        CallGlobal<T, T::HasAbility::kGlobalCache>::endJob(m_global.get());
      }
      void setupRun(EDProducerBase* iProd, RunIndex iIndex) override final {
        MyGlobal::set(iProd,m_global.get());
        MyGlobalRun::set(iProd, m_runs[iIndex].get());
      }
      void streamEndRunSummary(EDProducerBase* iProd,
                               edm::Run const& iRun,
                               edm::EventSetup const& iES) override final {
        auto s = m_runSummaries[iRun.index()].get();
        MyGlobalRunSummary::streamEndRunSummary(iProd,iRun,iES,s);
      }
 
      void setupLuminosityBlock(EDProducerBase* iProd, LuminosityBlockIndex iIndex) override final
      {
        MyGlobalLuminosityBlock::set(iProd, m_lumis[iIndex].get());
      }
      void streamEndLuminosityBlockSummary(EDProducerBase* iProd,
                                           edm::LuminosityBlock const& iLumi,
                                           edm::EventSetup const& iES) override final {
        auto s = m_lumiSummaries[iLumi.index()].get();
        MyGlobalLuminosityBlockSummary::streamEndLuminosityBlockSummary(iProd,iLumi,iES,s);
      }

      void doBeginRun(RunPrincipal& rp,
                      EventSetup const& c,
                      CurrentProcessingContext const* cpc)override final {
        if(T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache or T::HasAbility::kBeginRunProducer) {
          Run r(rp, moduleDescription());
          r.setConsumer(this);
          Run const& cnstR = r;
          RunIndex ri = rp.index();
          MyGlobalRun::beginRun(cnstR,c,m_global.get(),m_runs[ri]);
          typename T::RunContext rc(m_runs[ri].get(),m_global.get());
          MyGlobalRunSummary::beginRun(cnstR,c,&rc,m_runSummaries[ri]);
          if(T::HasAbility::kBeginRunProducer) {
            MyBeginRunProduce::produce(r,c,&rc);
            commit(r);
          }
        }
      }
      void doEndRun(RunPrincipal& rp,
                    EventSetup const& c,
                    CurrentProcessingContext const* cpc)override final
      {
        if(T::HasAbility::kRunCache or T::HasAbility::kRunSummaryCache or T::HasAbility::kEndRunProducer) {
          
          Run r(rp, moduleDescription());
          r.setConsumer(this);

          RunIndex ri = rp.index();
          typename T::RunContext rc(m_runs[ri].get(),m_global.get());
          if(T::HasAbility::kBeginRunProducer) {
            MyEndRunProduce::produce(r,c,&rc,m_runSummaries[ri].get());
            commit(r);
          }
          MyGlobalRunSummary::globalEndRun(r,c,&rc,m_runSummaries[ri].get());
          MyGlobalRun::endRun(r,c,&rc);
        }
      }

      void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                  CurrentProcessingContext const* cpc)override final
      {
        if(T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache or T::HasAbility::kBeginLuminosityBlockProducer) {
          LuminosityBlock lb(lbp, moduleDescription());
          lb.setConsumer(this);
          LuminosityBlock const& cnstLb = lb;
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          MyGlobalLuminosityBlock::beginLuminosityBlock(cnstLb,c,m_global.get(),m_lumis[li]);
          typename T::LuminosityBlockContext lc(m_lumis[li].get(),m_runs[ri].get(),m_global.get());
          MyGlobalLuminosityBlockSummary::beginLuminosityBlock(cnstLb,c,&lc,m_lumiSummaries[li]);
          if(T::HasAbility::kBeginLuminosityBlockProducer) {
            MyBeginLuminosityBlockProduce::produce(lb,c,&lc);
            commit(lb);
          }
        }
        
      }
      void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp,
                                EventSetup const& c,
                                CurrentProcessingContext const* cpc)override final {
        if(T::HasAbility::kLuminosityBlockCache or T::HasAbility::kLuminosityBlockSummaryCache or T::HasAbility::kEndLuminosityBlockProducer) {
          
          LuminosityBlock lb(lbp, moduleDescription());
          lb.setConsumer(this);
          
          LuminosityBlockIndex li = lbp.index();
          RunIndex ri = lbp.runPrincipal().index();
          typename T::LuminosityBlockContext lc(m_lumis[li].get(),m_runs[ri].get(),m_global.get());
          if(T::HasAbility::kBeginLuminosityBlockProducer) {
            MyEndLuminosityBlockProduce::produce(lb,c,&lc,m_lumiSummaries[li].get());
            commit(lb);
          }
          MyGlobalLuminosityBlockSummary::globalEndLuminosityBlock(lb,c,&lc,m_lumiSummaries[li].get());
          MyGlobalLuminosityBlock::endLuminosityBlock(lb,c,&lc);
        }
      }

      EDProducerAdaptor(const EDProducerAdaptor&); // stop default
      
      const EDProducerAdaptor& operator=(const EDProducerAdaptor&); // stop default
      
      // ---------- member data --------------------------------
      typename impl::choose_unique_ptr<typename T::GlobalCache>::type m_global;
      typename impl::choose_shared_vec<typename T::RunCache>::type m_runs;
      typename impl::choose_shared_vec<typename T::LuminosityBlockCache>::type m_lumis;
      typename impl::choose_shared_vec<typename T::RunSummaryCache>::type m_runSummaries;
      typename impl::choose_shared_vec<typename T::LuminosityBlockSummaryCache>::type m_lumiSummaries;
    };
  }
}

#endif
