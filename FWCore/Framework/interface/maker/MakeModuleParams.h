#ifndef FWCore_Framework_MakeModuleParams_h
#define FWCore_Framework_MakeModuleParams_h

/** ----------------------

This struct is used to communication parameters into the module factory.

---------------------- **/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <string>

namespace edm {
  class ProcessConfiguration;
  class SignallingProductRegistryFiller;
  class PreallocationConfiguration;

  struct MakeModuleParams {
    MakeModuleParams() : pset_(nullptr), reg_(nullptr), preallocate_(nullptr), processConfiguration_() {}

    MakeModuleParams(ParameterSet* pset,
                     SignallingProductRegistryFiller& reg,
                     PreallocationConfiguration const* prealloc,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration)
        : pset_(pset), reg_(&reg), preallocate_(prealloc), processConfiguration_(processConfiguration) {}

    ParameterSet* pset_;
    SignallingProductRegistryFiller* reg_;
    PreallocationConfiguration const* preallocate_;
    std::shared_ptr<ProcessConfiguration const> processConfiguration_;
  };
}  // namespace edm

#endif
