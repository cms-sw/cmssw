#ifndef FWCore_Framework_callAbilities_h
#define FWCore_Framework_callAbilities_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     callAbilities
// 
/**\class callAbilities callAbilities.h "FWCore/Framework/interface/streams/callAbilities.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat, 03 Aug 2013 16:28:35 GMT
//

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/stream/dummy_helpers.h"

// forward declarations
namespace edm {
  class Run;
  class EventSetup;
  namespace stream {
    //********************************
    // CallGlobal
    //********************************
    template<typename T , bool>
    struct CallGlobal {
      static void set(void* iProd,
                      typename T::GlobalCache const* iCache) {
        reinterpret_cast<T*>(iProd)->setGlobalCache(iCache);
      }
      static void endJob(typename T::GlobalCache* iCache) {
        T::globalEndJob(iCache);
      }
    };
    template<typename T>
    struct CallGlobal<T,false> {
      static void set(void* iProd,
                      void const* iCache) {}
      static void endJob(void* iCache) {
      }
    };
    //********************************
    // CallGlobalRun
    //********************************
    template<typename T, bool>
    struct CallGlobalRun {
      static void beginRun(edm::Run const& iRun,
                           edm::EventSetup const& iES,
                           typename T::GlobalContext const* iGC,
                           std::shared_ptr<typename T::RunCache>& oCache) {
        oCache = T::globalBeginRun(iRun,iES,iGC);
      }
    
      static void set(EDProducerBase* iProd, typename T::RunCache const* iCache) {
        static_cast<T>(iProd)->setRunCache(iCache);
      }
      
      static void globalEndRun(edm::Run const& iRun,
                               edm::EventSetup const& iES,
                               typename T::RunContext const* iContext) {
        T::globalEndRun(iRun, iES, iContext);
      }
    };

    template<typename T>
    struct CallGlobalRun<T,false> {
      static void beginRun(edm::Run const& ,
                           edm::EventSetup const& ,
                           typename T::GlobalCache const* ,
                           impl::dummy_ptr ) {
      }
      static void set(void* iProd,
                      typename T::RunCache const* iCache) {}
      static void globalEndRun(edm::Run const& ,
                               edm::EventSetup const& ,
                               typename T::RunContext const* ) {
      }
    };

    //********************************
    // CallGlobalRunSummary
    //********************************
    template<typename T, bool>
    struct CallGlobalRunSummary {
      static void beginRun(edm::Run const& iRun,
                           edm::EventSetup const& iES,
                           typename T::RunContext const* iRC,
                           std::shared_ptr<typename T::RunSummaryCache>& oCache) {
        oCache = T::globalBeginRunSummary(iRun,iES,iRC);
      }
      
      static void streamEndRunSummary(EDProducerBase* iProd, typename T::RunSummaryCache const* iCache) {
        static_cast<T>(iProd)->endRunSummary(iCache);
      }
      
      static void globalEndRun(edm::Run const& iRun,
                               edm::EventSetup const& iES,
                               typename T::RunContext const* iContext,
                               typename T::RunSummaryCache* iCache) {
        T::globalEndRunSummary(iRun, iES, iContext,iCache);
      }
    };
    
    template<typename T>
    struct CallGlobalRunSummary<T,false> {
      static void beginRun(edm::Run const& ,
                           edm::EventSetup const& ,
                           typename T::RunContext const* ,
                           impl::dummy_ptr ) {
      }
      static void streamEndRunSummary(EDProducerBase* iProd, typename T::RunSummaryCache const* iCache) {}
      static void globalEndRun(edm::Run const& ,
                               edm::EventSetup const& ,
                               typename T::RunContext const*,
                               typename T::RunSummaryCache* ) {
      }
    };

    //********************************
    // CallBeginRunProduce
    //********************************
    template<typename T, bool >
    struct CallBeginRunProduce {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC) {
        T::globalBeginRunProduce(iRun,iES,iRC);
      }
    };
    
    template<typename T>
    struct CallBeginRunProduce<T,false> {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC) {
      }
    };

    //********************************
    // CallEndRunProduce
    //********************************
    template<typename T, bool bProduce, bool bSummary>
    struct CallEndRunProduce {
      static void produce(edm::Run&,
                          edm::EventSetup const&,
                          typename T::RunContext const*,
                          typename T::RunSummaryCache const*) {}
    };

    template<typename T>
    struct CallEndRunProduce<T,true,false> {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC,
                          typename T::RunSummaryCache const*) {
        T::globalEndRunProduce(iRun,iES,iRC);
      }
    };

    template<typename T>
    struct CallEndRunProduce<T,true,true> {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC,
                          typename T::RunSummaryCache const* iRS) {
        T::globalEndRunProduce(iRun,iES,iRC, iRS);
      }
    };
}
}


#endif
