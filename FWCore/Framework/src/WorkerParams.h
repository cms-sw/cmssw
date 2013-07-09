#ifndef Framework_WorkerParams_h
#define Framework_WorkerParams_h

/** ----------------------

This struct is used to communication parameters into the worker factory.

---------------------- **/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/shared_ptr.hpp"

#include <string>

namespace edm {
  class ProcessConfiguration;
  class ProductRegistry;
  class ActionTable;

  struct WorkerParams {
    WorkerParams() :
      procPset_(nullptr), pset_(nullptr), reg_(nullptr), processConfiguration_(), actions_(nullptr)
      {}

    WorkerParams(ParameterSet const& procPset,
                 ParameterSet* pset,
                 ProductRegistry& reg,
                 boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                 ActionTable const& actions) :
      procPset_(&procPset),
      pset_(pset),
      reg_(&reg),
      processConfiguration_(processConfiguration),
      actions_(&actions) {}

    ParameterSet const* procPset_;
    ParameterSet* pset_;
    ProductRegistry* reg_;
    boost::shared_ptr<ProcessConfiguration const> processConfiguration_;
    ActionTable const* actions_;
  };
}

#endif
