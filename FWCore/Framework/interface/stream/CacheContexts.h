#ifndef FWCore_Framework_stream_CacheContexts_h
#define FWCore_Framework_stream_CacheContexts_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     CacheContexts
//
/**\class edm::stream::CacheContexts CacheContexts.h "FWCore/Framework/interface/stream/CacheContexts.h"

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
#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations
namespace edm {
  namespace stream {
    namespace impl {
      struct Last {};

      template <typename T, typename... U>
      struct AbilityToCache : public AbilityToCache<U...> {};

      template <typename G, typename... U>
      struct AbilityToCache<GlobalCache<G>, U...> : public AbilityToCache<U...> {
        using GlobalCache = G;
      };

      template <typename... CacheTypes, typename... U>
      struct AbilityToCache<InputProcessBlockCache<CacheTypes...>, U...> : public AbilityToCache<U...> {
        using InputProcessBlockCache = edm::impl::InputProcessBlockCacheImpl<CacheTypes...>;
      };

      template <typename R, typename... U>
      struct AbilityToCache<RunCache<R>, U...> : public AbilityToCache<U...> {
        using RunCache = R;
      };

      template <typename L, typename... U>
      struct AbilityToCache<LuminosityBlockCache<L>, U...> : public AbilityToCache<U...> {
        using LuminosityBlockCache = L;
      };

      template <typename R, typename... U>
      struct AbilityToCache<RunSummaryCache<R>, U...> : public AbilityToCache<U...> {
        using RunSummaryCache = R;
      };

      template <typename L, typename... U>
      struct AbilityToCache<LuminosityBlockSummaryCache<L>, U...> : public AbilityToCache<U...> {
        using LuminosityBlockSummaryCache = L;
      };

      template <>
      struct AbilityToCache<Last> {
        using GlobalCache = void;
        using InputProcessBlockCache = void;
        using RunCache = void;
        using LuminosityBlockCache = void;
        using RunSummaryCache = void;
        using LuminosityBlockSummaryCache = void;
      };

    }  // namespace impl
    template <typename... T>
    struct CacheContexts : public impl::AbilityToCache<T..., impl::Last> {};
  }  // namespace stream
}  // namespace edm

#endif
