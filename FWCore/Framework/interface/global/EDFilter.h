#ifndef FWCore_Framework_global_EDFilter_h
#define FWCore_Framework_global_EDFilter_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::global::EDFilter
// 
/**\class edm::global::EDFilter EDFilter.h "FWCore/Framework/interface/global/EDFilter.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 23 Jul 2013 11:51:07 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/filterAbilityToImplementor.h"
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace global {
    template< typename... T>
    class EDFilter : 
              public virtual EDFilterBase,
              public filter::SpecializeAbilityToImplementor<
        CheckAbility<edm::module::Abilities::kRunSummaryCache,T...>::kHasIt & CheckAbility<edm::module::Abilities::kEndRunProducer,T...>::kHasIt,
        CheckAbility<edm::module::Abilities::kLuminosityBlockSummaryCache,T...>::kHasIt & CheckAbility<edm::module::Abilities::kEndLuminosityBlockProducer,T...>::kHasIt,
        T>::Type...
    {
      
    public:
      EDFilter() = default;
// We do this only in the case of the intel compiler as this might
// end up creating a lot of code bloat due to inline symbols being generated 
// in each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~EDFilter() = default;
#endif
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    private:
      EDFilter(const EDFilter&) = delete;
      
      const EDFilter& operator=(const EDFilter&) = delete;
      
      // ---------- member data --------------------------------
      
    };

  }
}

#endif
