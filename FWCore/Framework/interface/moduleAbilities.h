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
// $Id: moduleAbilities.h,v 1.1 2013/05/17 14:49:43 chrjones Exp $
//

// system include files
#include "boost/mpl/if.hpp"

// user include files
#include "FWCore/Framework/interface/moduleAbilityEnums.h"

// forward declarations

namespace edm {
  namespace module {
    //Used in the case where ability is not available
    struct Empty{};
  }
  
  template<typename T>
  struct GlobalCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kGlobalCache;
    typedef T Type;
  };
  
  template<typename T>
  struct StreamCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kStreamCache;
    typedef T Type;
  };
  
  template<typename T>
  struct RunCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kRunCache;
    typedef T Type;
  };
  
  template<typename T>
  struct LuminosityBlockCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kLuminosityBlockCache;
    typedef T Type;
  };
  
  template<typename T>
  struct RunSummaryCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kRunSummaryCache;
    typedef T Type;
  };
  
  template<typename T>
  struct LuminosityBlockSummaryCache {
    static constexpr module::Abilities kAbilities=module::Abilities::kLuminosityBlockSummaryCache;
    typedef T Type;
  };
  
  struct BeginRunProducer {
    static constexpr module::Abilities kAbilities=module::Abilities::kBeginRunProducer;
    typedef module::Empty Type;
  };

  struct EndRunProducer {
    static constexpr module::Abilities kAbilities=module::Abilities::kEndRunProducer;
    typedef module::Empty Type;
  };

  struct BeginLuminosityBlockProducer {
    static constexpr module::Abilities kAbilities=module::Abilities::kBeginLuminosityBlockProducer;
    typedef module::Empty Type;
  };
  
  struct EndLuminosityBlockProducer {
    static constexpr module::Abilities kAbilities=module::Abilities::kEndLuminosityBlockProducer;
    typedef module::Empty Type;
  };

  
  //Recursively checks VArgs template arguments looking for the ABILITY
  template<module::Abilities ABILITY, typename... VArgs> struct CheckAbility;
  
  template<module::Abilities ABILITY, typename T, typename... VArgs>
  struct CheckAbility<ABILITY,T,VArgs...> {
    static constexpr bool kHasIt = (T::kAbilities==ABILITY) | CheckAbility<ABILITY,VArgs...>::kHasIt;
    typedef typename boost::mpl::if_c<(T::kAbilities==ABILITY),
    typename T::Type,
    typename CheckAbility<ABILITY,VArgs...>::Type>::type Type;
  };
  
  //End of the recursion
  template<module::Abilities ABILITY>
  struct CheckAbility<ABILITY> {
    static constexpr bool kHasIt=false;
    typedef edm::module::Empty Type;
  };

}

#endif
