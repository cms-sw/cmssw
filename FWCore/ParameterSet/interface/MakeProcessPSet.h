#ifndef PARAMETERSET_MAKE_RPOC_PSET_HHPP
#define PARAMETERSET_MAKE_RPOC_PSET_HHPP

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/shared_ptr.hpp"
#include <string>

namespace edm {
  boost::shared_ptr<edm::ParameterSet> makeProcessPSet(const std::string& config);
}

#endif
