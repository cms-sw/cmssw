#ifndef FWCore_Framework_stream_CacheContexts_h
#define FWCore_Framework_stream_CacheContexts_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     CacheContexts
// 
/**\class CacheContexts CacheContexts.h "FWCore/Framework/interface/stream/CacheContexts.h"

 Description: Helper class used to identify the caches requested by a module

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 00:17:38 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations
namespace edm {
  namespace stream {
    namespace impl {
      struct Last {};
      
      template<typename T, typename... U>
      struct AbilityToCache : public AbilityToCache<U...> {};
      
      template<typename G, typename... U>
      struct AbilityToCache<GlobalCache<G>, U...> : public AbilityToCache<U...> {
        typedef G GlobalCache;
      };
      
      template<typename R, typename... U>
      struct AbilityToCache<RunCache<R>, U...> : public AbilityToCache<U...> {
        typedef R RunCache;
      };
      
      template<typename L, typename... U>
      struct AbilityToCache<LuminosityBlockCache<L>, U...> : public AbilityToCache<U...> {
        typedef L LuminosityBlockCache;
      };

      template<typename R, typename... U>
      struct AbilityToCache<RunSummaryCache<R>, U...> : public AbilityToCache<U...> {
        typedef R RunSummaryCache;
      };
      
      template<typename L, typename... U>
      struct AbilityToCache<LuminosityBlockSummaryCache<L>, U...> : public AbilityToCache<U...> {
        typedef L LuminosityBlockSummaryCache;
      };
      
      template<>
      struct AbilityToCache<Last> {
        typedef void GlobalCache ;
        typedef void RunCache ;
        typedef void LuminosityBlockCache ;
        typedef void RunSummaryCache ;
        typedef void LuminosityBlockSummaryCache ;
      };
      
    }
    template <typename... T>
    struct CacheContexts : public impl::AbilityToCache<T...,impl::Last>   {
    };
  }
}


#endif
