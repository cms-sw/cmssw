#ifndef FWK_WORKER_PARAMS_H
#define FWK_WORKER_PARAMS_H

/** ----------------------

This struct is used to communication parameters into the worker factory.

---------------------- **/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edm
{
  class ProductRegistry;
  class ActionTable;

  struct WorkerParams
  {
    WorkerParams(): 
      pset_(),reg_(),actions_(),
      processName_(),versionNumber__(),pass_() { }

    WorkerParams(ParameterSet const& pset,
		 ProductRegistry& reg,
		 ActionTable& actions,
		 const std::string& pn,
		 unsigned long vn=0, unsigned long pass=0):
      pset_(&pset),reg_(&reg),actions_(&actions),
      processName_(pn),versionNumber__(vn),pass_(pass) { }

    const ParameterSet* pset_;
    ProductRegistry* reg_;
    ActionTable* actions_;
    std::string processName_;
    unsigned long versionNumber__;
    unsigned long pass_;
  };
}

#endif
