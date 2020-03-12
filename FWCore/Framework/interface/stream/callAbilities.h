#ifndef FWCore_Framework_stream_callAbilities_h
#define FWCore_Framework_stream_callAbilities_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     callAbilities
//
/**\class callAbilities callAbilities.h "FWCore/Framework/interface/stream/callAbilities.h"

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
  class LuminosityBlock;
  namespace stream {
    //********************************
    // CallGlobal
    //********************************
    template <typename T, bool>
    struct CallGlobalImpl {
      template <typename B>
      static void set(B* iProd, typename T::GlobalCache const* iCache) {
        static_cast<T*>(iProd)->setGlobalCache(iCache);
      }
      static void endJob(typename T::GlobalCache* iCache) { T::globalEndJob(iCache); }
    };
    template <typename T>
    struct CallGlobalImpl<T, false> {
      static void set(void* iProd, void const* iCache) {}
      static void endJob(void* iCache) {}
    };

    template <typename T>
    using CallGlobal = CallGlobalImpl<T, T::HasAbility::kGlobalCache>;
    //********************************
    // CallGlobalRun
    //********************************
    template <typename T, bool>
    struct CallGlobalRunImpl {
      static void beginRun(edm::Run const& iRun,
                           edm::EventSetup const& iES,
                           typename T::GlobalCache const* iGC,
                           std::shared_ptr<typename T::RunCache const>& oCache) {
        oCache = T::globalBeginRun(iRun, iES, iGC);
      }

      template <typename B>
      static void set(B* iProd, typename T::RunCache const* iCache) {
        static_cast<T*>(iProd)->setRunCache(iCache);
      }

      static void endRun(edm::Run const& iRun, edm::EventSetup const& iES, typename T::RunContext const* iContext) {
        T::globalEndRun(iRun, iES, iContext);
      }
    };

    template <typename T>
    struct CallGlobalRunImpl<T, false> {
      static void beginRun(edm::Run const&, edm::EventSetup const&, typename T::GlobalCache const*, impl::dummy_ptr) {}
      static void set(void* iProd, typename T::RunCache const* iCache) {}
      static void endRun(edm::Run const&, edm::EventSetup const&, typename T::RunContext const*) {}
    };

    template <typename T>
    using CallGlobalRun = CallGlobalRunImpl<T, T::HasAbility::kRunCache>;

    //********************************
    // CallGlobalRunSummary
    //********************************
    template <typename T, bool>
    struct CallGlobalRunSummaryImpl {
      static void beginRun(edm::Run const& iRun,
                           edm::EventSetup const& iES,
                           typename T::RunContext const* iRC,
                           std::shared_ptr<typename T::RunSummaryCache>& oCache) {
        oCache = T::globalBeginRunSummary(iRun, iES, iRC);
      }
      template <typename B>
      static void streamEndRunSummary(B* iProd,
                                      edm::Run const& iRun,
                                      edm::EventSetup const& iES,
                                      typename T::RunSummaryCache* iCache) {
        static_cast<T*>(iProd)->endRunSummary(iRun, iES, iCache);
      }

      static void globalEndRun(edm::Run const& iRun,
                               edm::EventSetup const& iES,
                               typename T::RunContext const* iContext,
                               typename T::RunSummaryCache* iCache) {
        T::globalEndRunSummary(iRun, iES, iContext, iCache);
      }
    };

    template <typename T>
    struct CallGlobalRunSummaryImpl<T, false> {
      static void beginRun(edm::Run const&, edm::EventSetup const&, typename T::RunContext const*, impl::dummy_ptr) {}
      static void streamEndRunSummary(void* iProd,
                                      edm::Run const&,
                                      edm::EventSetup const&,
                                      typename T::RunSummaryCache const* iCache) {}
      static void globalEndRun(edm::Run const&,
                               edm::EventSetup const&,
                               typename T::RunContext const*,
                               typename T::RunSummaryCache*) {}
    };

    template <typename T>
    using CallGlobalRunSummary = CallGlobalRunSummaryImpl<T, T::HasAbility::kRunSummaryCache>;

    //********************************
    // CallGlobalLuminosityBlock
    //********************************
    template <typename T, bool>
    struct CallGlobalLuminosityBlockImpl {
      static void beginLuminosityBlock(edm::LuminosityBlock const& Lumi,
                                       edm::EventSetup const& iES,
                                       typename T::RunContext const* iRC,
                                       std::shared_ptr<typename T::LuminosityBlockCache const>& oCache) {
        oCache = T::globalBeginLuminosityBlock(Lumi, iES, iRC);
      }

      template <typename B>
      static void set(B* iProd, typename T::LuminosityBlockCache const* iCache) {
        static_cast<T*>(iProd)->setLuminosityBlockCache(iCache);
      }

      static void endLuminosityBlock(edm::LuminosityBlock const& Lumi,
                                     edm::EventSetup const& iES,
                                     typename T::LuminosityBlockContext const* iContext) {
        T::globalEndLuminosityBlock(Lumi, iES, iContext);
      }
    };

    template <typename T>
    struct CallGlobalLuminosityBlockImpl<T, false> {
      static void beginLuminosityBlock(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       typename T::RunContext const*,
                                       impl::dummy_ptr) {}
      static void set(void* iProd, typename T::LuminosityBlockCache const* iCache) {}
      static void endLuminosityBlock(edm::LuminosityBlock const&,
                                     edm::EventSetup const&,
                                     typename T::LuminosityBlockContext const*) {}
    };
    template <typename T>
    using CallGlobalLuminosityBlock = CallGlobalLuminosityBlockImpl<T, T::HasAbility::kLuminosityBlockCache>;

    //********************************
    // CallGlobalLuminosityBlockSummary
    //********************************
    template <typename T, bool>
    struct CallGlobalLuminosityBlockSummaryImpl {
      static void beginLuminosityBlock(edm::LuminosityBlock const& Lumi,
                                       edm::EventSetup const& iES,
                                       typename T::LuminosityBlockContext const* iRC,
                                       std::shared_ptr<typename T::LuminosityBlockSummaryCache>& oCache) {
        oCache = T::globalBeginLuminosityBlockSummary(Lumi, iES, iRC);
      }
      template <typename B>
      static void streamEndLuminosityBlockSummary(B* iProd,
                                                  edm::LuminosityBlock const& iLumi,
                                                  edm::EventSetup const& iES,
                                                  typename T::LuminosityBlockSummaryCache* iCache) {
        static_cast<T*>(iProd)->endLuminosityBlockSummary(iLumi, iES, iCache);
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& Lumi,
                                           edm::EventSetup const& iES,
                                           typename T::LuminosityBlockContext const* iContext,
                                           typename T::LuminosityBlockSummaryCache* iCache) {
        T::globalEndLuminosityBlockSummary(Lumi, iES, iContext, iCache);
      }
    };

    template <typename T>
    struct CallGlobalLuminosityBlockSummaryImpl<T, false> {
      static void beginLuminosityBlock(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       typename T::LuminosityBlockContext const*,
                                       impl::dummy_ptr) {}
      static void streamEndLuminosityBlockSummary(void* iProd,
                                                  edm::LuminosityBlock const&,
                                                  edm::EventSetup const&,
                                                  typename T::LuminosityBlockSummaryCache* iCache) {}
      static void globalEndLuminosityBlock(edm::LuminosityBlock const&,
                                           edm::EventSetup const&,
                                           typename T::LuminosityBlockContext const*,
                                           typename T::LuminosityBlockSummaryCache*) {}
    };

    template <typename T>
    using CallGlobalLuminosityBlockSummary =
        CallGlobalLuminosityBlockSummaryImpl<T, T::HasAbility::kLuminosityBlockSummaryCache>;

    //********************************
    // CallBeginRunProduce
    //********************************
    template <typename T, bool>
    struct CallBeginRunProduceImpl {
      static void produce(edm::Run& iRun, edm::EventSetup const& iES, typename T::RunContext const* iRC) {
        T::globalBeginRunProduce(iRun, iES, iRC);
      }
    };

    template <typename T>
    struct CallBeginRunProduceImpl<T, false> {
      static void produce(edm::Run& iRun, edm::EventSetup const& iES, typename T::RunContext const* iRC) {}
    };

    template <typename T>
    using CallBeginRunProduce = CallBeginRunProduceImpl<T, T::HasAbility::kBeginRunProducer>;

    //********************************
    // CallEndRunProduce
    //********************************
    template <typename T, bool bProduce, bool bSummary>
    struct CallEndRunProduceImpl {
      static void produce(edm::Run&,
                          edm::EventSetup const&,
                          typename T::RunContext const*,
                          typename T::RunSummaryCache const*) {}
    };

    template <typename T>
    struct CallEndRunProduceImpl<T, true, false> {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC,
                          typename T::RunSummaryCache const*) {
        T::globalEndRunProduce(iRun, iES, iRC);
      }
    };

    template <typename T>
    struct CallEndRunProduceImpl<T, true, true> {
      static void produce(edm::Run& iRun,
                          edm::EventSetup const& iES,
                          typename T::RunContext const* iRC,
                          typename T::RunSummaryCache const* iRS) {
        T::globalEndRunProduce(iRun, iES, iRC, iRS);
      }
    };

    template <typename T>
    using CallEndRunProduce = CallEndRunProduceImpl<T, T::HasAbility::kEndRunProducer, T::HasAbility::kRunSummaryCache>;

    //********************************
    // CallBeginLuminosityBlockProduce
    //********************************
    template <typename T, bool>
    struct CallBeginLuminosityBlockProduceImpl {
      static void produce(edm::LuminosityBlock& Lumi,
                          edm::EventSetup const& iES,
                          typename T::LuminosityBlockContext const* iRC) {
        T::globalBeginLuminosityBlockProduce(Lumi, iES, iRC);
      }
    };

    template <typename T>
    struct CallBeginLuminosityBlockProduceImpl<T, false> {
      static void produce(edm::LuminosityBlock& Lumi,
                          edm::EventSetup const& iES,
                          typename T::LuminosityBlockContext const* iRC) {}
    };

    template <typename T>
    using CallBeginLuminosityBlockProduce =
        CallBeginLuminosityBlockProduceImpl<T, T::HasAbility::kBeginLuminosityBlockProducer>;

    //********************************
    // CallEndLuminosityBlockProduce
    //********************************
    template <typename T, bool bProduce, bool bSummary>
    struct CallEndLuminosityBlockProduceImpl {
      static void produce(edm::LuminosityBlock&,
                          edm::EventSetup const&,
                          typename T::LuminosityBlockContext const*,
                          typename T::LuminosityBlockSummaryCache const*) {}
    };

    template <typename T>
    struct CallEndLuminosityBlockProduceImpl<T, true, false> {
      static void produce(edm::LuminosityBlock& Lumi,
                          edm::EventSetup const& iES,
                          typename T::LuminosityBlockContext const* iRC,
                          typename T::LuminosityBlockSummaryCache const*) {
        T::globalEndLuminosityBlockProduce(Lumi, iES, iRC);
      }
    };

    template <typename T>
    struct CallEndLuminosityBlockProduceImpl<T, true, true> {
      static void produce(edm::LuminosityBlock& Lumi,
                          edm::EventSetup const& iES,
                          typename T::LuminosityBlockContext const* iRC,
                          typename T::LuminosityBlockSummaryCache const* iRS) {
        T::globalEndLuminosityBlockProduce(Lumi, iES, iRC, iRS);
      }
    };

    template <typename T>
    using CallEndLuminosityBlockProduce =
        CallEndLuminosityBlockProduceImpl<T,
                                          T::HasAbility::kEndLuminosityBlockProducer,
                                          T::HasAbility::kLuminosityBlockSummaryCache>;

  }  // namespace stream
}  // namespace edm

#endif
