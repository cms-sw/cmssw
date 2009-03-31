#ifndef Framework_WorkerParams_h
#define Framework_WorkerParams_h

/** ----------------------

This struct is used to communication parameters into the worker factory.

---------------------- **/

#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include "boost/shared_ptr.hpp"

namespace edm
{
  class ProductRegistry;
  class ActionTable;

  struct WorkerParams
  {
    WorkerParams(): 
      procPset_(0), pset_(0), reg_(0), processConfiguration_(), actions_(0)
      {}

    WorkerParams(ParameterSet const& procPset,
		 ParameterSet * pset,
                 ProductRegistry& reg,
		 boost::shared_ptr<ProcessConfiguration> processConfiguration,
		 ActionTable& actions) :
      procPset_(&procPset),
      pset_(pset),
      reg_(&reg),
      processConfiguration_(processConfiguration),
      actions_(&actions) {}

    ParameterSet const* procPset_;
    ParameterSet* pset_;
    ProductRegistry* reg_;
    boost::shared_ptr<ProcessConfiguration> processConfiguration_;
    ActionTable* actions_;
  };
}

#endif
