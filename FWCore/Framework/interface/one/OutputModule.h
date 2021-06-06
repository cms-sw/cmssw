#ifndef FWCore_Framework_one_OutputModule_h
#define FWCore_Framework_one_OutputModule_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::one::OutputModule
//
/**\class edm::one::OutputModule OutputModule.h "FWCore/Framework/interface/one/OutputModule.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 18:21:29 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/outputmoduleAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace one {
    template <typename... T>
    class OutputModule : public virtual OutputModuleBase, public outputmodule::AbilityToImplementor<T>::Type... {
    public:
      OutputModule(edm::ParameterSet const& iPSet)
          : OutputModuleBase(iPSet), outputmodule::AbilityToImplementor<T>::Type(iPSet)... {}
      // Required to work around ICC bug, but possible source of bloat in gcc.
      // We do this only in the case of the intel compiler as this might end up
      // creating a lot of code bloat due to inline symbols being generated in
      // each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~OutputModule() = default;
#endif

      // ---------- const member functions ---------------------
      bool wantsProcessBlocks() const final { return WantsProcessBlockTransitions<T...>::value; }
      bool wantsInputProcessBlocks() const final { return WantsInputProcessBlockTransitions<T...>::value; }
      bool wantsGlobalRuns() const final { return WantsGlobalRunTransitions<T...>::value; }
      bool wantsGlobalLuminosityBlocks() const final { return WantsGlobalLuminosityBlockTransitions<T...>::value; }

      SerialTaskQueue* globalRunsQueue() final { return globalRunsQueue_.queue(); }
      SerialTaskQueue* globalLuminosityBlocksQueue() final { return globalLuminosityBlocksQueue_.queue(); }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      OutputModule(const OutputModule&) = delete;  // stop default

      const OutputModule& operator=(const OutputModule&) = delete;  // stop default

      // ---------- member data --------------------------------
      impl::OptionalSerialTaskQueueHolder<WantsSerialGlobalRunTransitions<T...>::value> globalRunsQueue_;
      impl::OptionalSerialTaskQueueHolder<WantsSerialGlobalLuminosityBlockTransitions<T...>::value>
          globalLuminosityBlocksQueue_;
    };
  }  // namespace one
}  // namespace edm

#endif
