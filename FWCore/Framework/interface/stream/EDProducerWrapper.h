#ifndef Subsystem_Package_EDProducerWrapper_h
#define Subsystem_Package_EDProducerWrapper_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     EDProducerWrapper
// 
/**\class EDProducerWrapper EDProducerWrapper.h "EDProducerWrapper.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:18 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducerWrapperBase.h"
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
    class EDProducerWrapper : public EDProducerWrapperBase
    {
      
    public:
      EDProducerWrapper( edm::ParameterSet const& iPSet)
      {
        m_runs.resize(1);
        m_lumis.resize(1);
        typename T::GlobalCache const* dummy=nullptr;
        m_global.reset( impl::makeGlobal<T>(iPSet,dummy).release());
        this->createStreamModules([this,&iPSet] () -> EDProducerBase* {return impl::makeStreamModule<T>(iPSet,m_global.get());});
      }
      ~EDProducerWrapper() {
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
      
      void doEndJob() override final {
        CallGlobal<T, T::HasAbility::kGlobalCache>::endJob(m_global.get());
      }
      void setupRun(EDProducerBase* iProd, RunIndex iIndex) override final {
        MyGlobal::set(iProd,m_global.get());
        MyGlobalRun::set(iProd, m_runs[iIndex].get());
      }
      void streamEndRunSummary(EDProducerBase*,edm::Run const&, edm::EventSetup const&) override final;

      void setupLuminosityBlock(EDProducerBase*, LuminosityBlockIndex) override final;
      void streamEndLuminosityBlockSummary(EDProducerBase*,edm::LuminosityBlock const&, edm::EventSetup const&) override final;

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
                    CurrentProcessingContext const* cpc)override final;
      void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                          CurrentProcessingContext const* cpc)override final;
      void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                        CurrentProcessingContext const* cpc)override final;

      EDProducerWrapper(const EDProducerWrapper&); // stop default
      
      const EDProducerWrapper& operator=(const EDProducerWrapper&); // stop default
      
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
