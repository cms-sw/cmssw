#ifndef FWCore_Framework_moduleAbilities_h
#define FWCore_Framework_moduleAbilities_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     moduleAbilities
//
/**\file moduleAbilities moduleAbilities.h "FWCore/Framework/interface/moduleAbilities.h"

 Description: Template arguments for stream::{Module}, global::{Module}, one::{Module} classes

 Usage:
    These classes are used the declare the 'abilities' a developer wants to make use of in their module.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 07 May 2013 19:19:53 GMT
//

// system include files
#include <type_traits>

// user include files
#include "FWCore/Framework/interface/moduleAbilityEnums.h"

// forward declarations

namespace edm {
  namespace module {
    //Used in the case where ability is not available
    struct Empty {};
  }  // namespace module

  template <typename T>
  struct GlobalCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kGlobalCache;
    typedef T Type;
  };

  template <typename T>
  struct StreamCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kStreamCache;
    typedef T Type;
  };

  template <typename... CacheTypes>
  struct InputProcessBlockCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kInputProcessBlockCache;
  };

  template <typename T>
  struct RunCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kRunCache;
    typedef T Type;
  };

  template <typename T>
  struct LuminosityBlockCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kLuminosityBlockCache;
    typedef T Type;
  };

  template <typename T>
  struct RunSummaryCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kRunSummaryCache;
    typedef T Type;
  };

  template <typename T>
  struct LuminosityBlockSummaryCache {
    static constexpr module::Abilities kAbilities = module::Abilities::kLuminosityBlockSummaryCache;
    typedef T Type;
  };

  struct WatchProcessBlock {
    static constexpr module::Abilities kAbilities = module::Abilities::kWatchProcessBlock;
    typedef module::Empty Type;
  };

  struct BeginProcessBlockProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kBeginProcessBlockProducer;
    typedef module::Empty Type;
  };

  struct EndProcessBlockProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kEndProcessBlockProducer;
    typedef module::Empty Type;
  };

  struct BeginRunProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kBeginRunProducer;
    typedef module::Empty Type;
  };

  struct EndRunProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kEndRunProducer;
    typedef module::Empty Type;
  };

  struct BeginLuminosityBlockProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kBeginLuminosityBlockProducer;
    typedef module::Empty Type;
  };

  struct EndLuminosityBlockProducer {
    static constexpr module::Abilities kAbilities = module::Abilities::kEndLuminosityBlockProducer;
    typedef module::Empty Type;
  };

  struct WatchInputFiles {
    static constexpr module::Abilities kAbilities = module::Abilities::kWatchInputFiles;
    typedef module::Empty Type;
  };

  struct ExternalWork {
    static constexpr module::Abilities kAbilities = module::Abilities::kExternalWork;
    typedef module::Empty Type;
  };

  struct Accumulator {
    static constexpr module::Abilities kAbilities = module::Abilities::kAccumulator;
    typedef module::Empty Type;
  };

  struct Transformer {
    static constexpr module::Abilities kAbilities = module::Abilities::kTransformer;
    using Type = module::Empty;
  };

  //Recursively checks VArgs template arguments looking for the ABILITY
  template <module::Abilities ABILITY, typename... VArgs>
  struct CheckAbility;

  template <module::Abilities ABILITY, typename T, typename... VArgs>
  struct CheckAbility<ABILITY, T, VArgs...> {
    static constexpr bool kHasIt = (T::kAbilities == ABILITY) | CheckAbility<ABILITY, VArgs...>::kHasIt;
  };

  //End of the recursion
  template <module::Abilities ABILITY>
  struct CheckAbility<ABILITY> {
    static constexpr bool kHasIt = false;
  };

  template <typename... VArgs>
  struct WantsProcessBlockTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kWatchProcessBlock, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kBeginProcessBlockProducer, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kEndProcessBlockProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct WantsInputProcessBlockTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kInputProcessBlockCache, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct WantsGlobalRunTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kRunCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kRunSummaryCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kBeginRunProducer, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kEndRunProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct WantsGlobalLuminosityBlockTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kLuminosityBlockCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kLuminosityBlockSummaryCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kBeginLuminosityBlockProducer, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kEndLuminosityBlockProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct WantsStreamRunTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kStreamCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kRunSummaryCache, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct WantsStreamLuminosityBlockTransitions {
    static constexpr bool value = CheckAbility<module::Abilities::kStreamCache, VArgs...>::kHasIt or
                                  CheckAbility<module::Abilities::kLuminosityBlockSummaryCache, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInBeginProcessBlocks {
    static constexpr bool value = CheckAbility<module::Abilities::kBeginProcessBlockProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInEndProcessBlocks {
    static constexpr bool value = CheckAbility<module::Abilities::kEndProcessBlockProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInBeginRuns {
    static constexpr bool value = CheckAbility<module::Abilities::kBeginRunProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInEndRuns {
    static constexpr bool value = CheckAbility<module::Abilities::kEndRunProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInBeginLumis {
    static constexpr bool value = CheckAbility<module::Abilities::kBeginLuminosityBlockProducer, VArgs...>::kHasIt;
  };

  template <typename... VArgs>
  struct HasAbilityToProduceInEndLumis {
    static constexpr bool value = CheckAbility<module::Abilities::kEndLuminosityBlockProducer, VArgs...>::kHasIt;
  };

}  // namespace edm

#endif
