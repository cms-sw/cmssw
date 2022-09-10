#ifndef FWCore_Framework_AbilityChecker_h
#define FWCore_Framework_AbilityChecker_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     AbilityChecker
//
/**\class edm::stream::AbilityChecker AbilityChecker.h "FWCore/Framework/interface/stream/AbilityChecker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat, 03 Aug 2013 15:38:02 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations
namespace edm {
  namespace stream {
    namespace impl {
      struct LastCheck {};

      template <typename T, typename... U>
      struct HasAbility;

      template <typename G, typename... U>
      struct HasAbility<GlobalCache<G>, U...> : public HasAbility<U...> {
        static constexpr bool kGlobalCache = true;
      };

      template <typename... CacheTypes, typename... U>
      struct HasAbility<InputProcessBlockCache<CacheTypes...>, U...> : public HasAbility<U...> {
        static constexpr bool kInputProcessBlockCache = true;
      };

      template <typename R, typename... U>
      struct HasAbility<RunCache<R>, U...> : public HasAbility<U...> {
        static constexpr bool kRunCache = true;
      };

      template <typename R, typename... U>
      struct HasAbility<LuminosityBlockCache<R>, U...> : public HasAbility<U...> {
        static constexpr bool kLuminosityBlockCache = true;
      };

      template <typename R, typename... U>
      struct HasAbility<RunSummaryCache<R>, U...> : public HasAbility<U...> {
        static constexpr bool kRunSummaryCache = true;
      };

      template <typename R, typename... U>
      struct HasAbility<LuminosityBlockSummaryCache<R>, U...> : public HasAbility<U...> {
        static constexpr bool kLuminosityBlockSummaryCache = true;
      };

      template <typename... U>
      struct HasAbility<edm::WatchProcessBlock, U...> : public HasAbility<U...> {
        static constexpr bool kWatchProcessBlock = true;
      };

      template <typename... U>
      struct HasAbility<edm::BeginProcessBlockProducer, U...> : public HasAbility<U...> {
        static constexpr bool kBeginProcessBlockProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::EndProcessBlockProducer, U...> : public HasAbility<U...> {
        static constexpr bool kEndProcessBlockProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::BeginRunProducer, U...> : public HasAbility<U...> {
        static constexpr bool kBeginRunProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::EndRunProducer, U...> : public HasAbility<U...> {
        static constexpr bool kEndRunProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::BeginLuminosityBlockProducer, U...> : public HasAbility<U...> {
        static constexpr bool kBeginLuminosityBlockProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::EndLuminosityBlockProducer, U...> : public HasAbility<U...> {
        static constexpr bool kEndLuminosityBlockProducer = true;
      };

      template <typename... U>
      struct HasAbility<edm::ExternalWork, U...> : public HasAbility<U...> {
        static constexpr bool kExternalWork = true;
      };

      template <typename... U>
      struct HasAbility<edm::Transformer, U...> : public HasAbility<U...> {
        static constexpr bool kTransformer = true;
      };

      template <typename... U>
      struct HasAbility<edm::Accumulator, U...> : public HasAbility<U...> {
        static constexpr bool kAccumulator = true;
      };

      template <>
      struct HasAbility<LastCheck> {
        static constexpr bool kGlobalCache = false;
        static constexpr bool kInputProcessBlockCache = false;
        static constexpr bool kRunCache = false;
        static constexpr bool kLuminosityBlockCache = false;
        static constexpr bool kRunSummaryCache = false;
        static constexpr bool kLuminosityBlockSummaryCache = false;
        static constexpr bool kWatchProcessBlock = false;
        static constexpr bool kBeginProcessBlockProducer = false;
        static constexpr bool kEndProcessBlockProducer = false;
        static constexpr bool kBeginRunProducer = false;
        static constexpr bool kEndRunProducer = false;
        static constexpr bool kBeginLuminosityBlockProducer = false;
        static constexpr bool kEndLuminosityBlockProducer = false;
        static constexpr bool kExternalWork = false;
        static constexpr bool kAccumulator = false;
        static constexpr bool kTransformer = false;
      };
    }  // namespace impl
    template <typename... T>
    struct AbilityChecker : public impl::HasAbility<T..., impl::LastCheck> {};
  }  // namespace stream
}  // namespace edm

#endif
