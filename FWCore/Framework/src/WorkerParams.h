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
      process_name_(),version_number_(),pass_() { }

    WorkerParams(ParameterSet const& pset,
		 ProductRegistry* reg,
		 ActionTable* actions,
		 const std::string& pn,
		 unsigned long vn=0, unsigned long pass=0):
      pset_(&pset),reg_(reg),actions_(actions),
      process_name_(pn),version_number_(vn),pass_(pass) { }

    const ParameterSet* pset_;
    ProductRegistry* reg_;
    ActionTable* actions_;
    std::string process_name_;
    unsigned long version_number_;
    unsigned long pass_;
  };
}

#endif
