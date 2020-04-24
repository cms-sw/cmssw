#ifndef FWCore_Framework_WorkerParams_h
#define FWCore_Framework_WorkerParams_h

/** ----------------------

This struct is used to communication parameters into the worker factory.

---------------------- **/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <string>

namespace edm {
  class ProcessConfiguration;
  class ProductRegistry;
  class ExceptionToActionTable;
  class PreallocationConfiguration;

  struct WorkerParams {
    WorkerParams() :
      pset_(nullptr), reg_(nullptr), preallocate_(nullptr),processConfiguration_(), actions_(nullptr)
      {}

    WorkerParams(ParameterSet* pset,
                 ProductRegistry& reg,
                 PreallocationConfiguration const* prealloc,
                 std::shared_ptr<ProcessConfiguration const> processConfiguration,
                 ExceptionToActionTable const& actions) :
      pset_(pset),
      reg_(&reg),
      preallocate_(prealloc),
      processConfiguration_(processConfiguration),
      actions_(&actions) {}

    ParameterSet* pset_;
    ProductRegistry* reg_;
    PreallocationConfiguration const* preallocate_;
    std::shared_ptr<ProcessConfiguration const> processConfiguration_;
    ExceptionToActionTable const* actions_;
  };
}

#endif
