#ifndef FWCore_Framework_one_moduleAbilities_h
#define FWCore_Framework_one_moduleAbilities_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     moduleAbilities
// 
/**\file moduleAbilities moduleAbilities.h "FWCore/Framework/interface/one/moduleAbilities.h"

 Description: Template arguments which only apply to one::{Module} classes

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 07 May 2013 19:38:53 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace one {
    struct SharedResources {
      static constexpr module::Abilities kAbilities=module::Abilities::kOneSharedResources;
      typedef module::Empty Type;
    };

    struct WatchRuns {
      static constexpr module::Abilities kAbilities=module::Abilities::kOneWatchRuns;
      typedef module::Empty Type;
    };

    struct WatchLuminosityBlocks {
      static constexpr module::Abilities kAbilities=module::Abilities::kOneWatchLuminosityBlocks;
      typedef module::Empty Type;
    };
    
    template<typename... VArgs>
    struct WantsGlobalRunTransitions {
      static constexpr bool value = CheckAbility<module::Abilities::kRunCache,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kOneWatchRuns,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kBeginRunProducer,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kEndRunProducer, VArgs...>::kHasIt;
    };
    
    template<typename... VArgs>
    struct WantsGlobalLuminosityBlockTransitions {
      static constexpr bool value = CheckAbility<module::Abilities::kLuminosityBlockCache,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kOneWatchLuminosityBlocks,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kBeginLuminosityBlockProducer,VArgs...>::kHasIt or
      CheckAbility<module::Abilities::kEndLuminosityBlockProducer, VArgs...>::kHasIt;
    };

    //RunCache and LuminosityBlockCache do not require serializing their
    // respective transitions
    template<typename... VArgs>
    struct WantsSerialGlobalRunTransitions {
      static constexpr bool value =
      WantsGlobalRunTransitions<VArgs...>::value and not CheckAbility<module::Abilities::kRunCache,VArgs...>::kHasIt;
    };

    template<typename... VArgs>
    struct WantsSerialGlobalLuminosityBlockTransitions {
      static constexpr bool value =
      WantsGlobalLuminosityBlockTransitions<VArgs...>::value and not CheckAbility<module::Abilities::kLuminosityBlockCache,VArgs...>::kHasIt;
    };
  }
}


#endif
