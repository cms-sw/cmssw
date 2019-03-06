#ifndef FWCore_Framework_global_EDAnalyzer_h
#define FWCore_Framework_global_EDAnalyzer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::global::EDAnalyzer
//
/**\class edm::global::EDAnalyzer EDAnalyzer.h "FWCore/Framework/interface/global/EDAnalyzer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:51:07 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/analyzerAbilityToImplementor.h"
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace global {
    template <typename... T>
    class EDAnalyzer : public analyzer::AbilityToImplementor<T>::Type..., public virtual EDAnalyzerBase {
    public:
      EDAnalyzer() = default;
// We do this only in the case of the intel compiler as this might
// end up creating a lot of code bloat due to inline symbols being generated
// in each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~EDAnalyzer() {}
#endif
      // ---------- const member functions ---------------------
      bool wantsGlobalRuns() const final { return WantsGlobalRunTransitions<T...>::value; }
      bool wantsGlobalLuminosityBlocks() const final { return WantsGlobalLuminosityBlockTransitions<T...>::value; }

      bool wantsStreamRuns() const final { return WantsStreamRunTransitions<T...>::value; }
      bool wantsStreamLuminosityBlocks() const final { return WantsStreamLuminosityBlockTransitions<T...>::value; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      EDAnalyzer(const EDAnalyzer&) = delete;

      const EDAnalyzer& operator=(const EDAnalyzer&) = delete;

      // ---------- member data --------------------------------
    };

  }  // namespace global
}  // namespace edm

#endif
