#ifndef ParametetSet_MakeProcessPSet_h
#define ParametetSet_MakeProcessPSet_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/shared_ptr.hpp"
#include <string>

namespace edm {
  boost::shared_ptr<edm::ParameterSet> makeProcessPSet(const std::string& config);
}

#endif
