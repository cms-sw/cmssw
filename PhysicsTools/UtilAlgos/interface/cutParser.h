#ifndef UtilAlgos_cutParset_h
#define UtilAlgos_cutParset_h
#include "PhysicsTools/UtilAlgos/interface/ReflexSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MethodMap.h"
#include <string>
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    typedef boost::shared_ptr<ReflexSelector> selector_ptr;
    bool cutParser( const std::string &, const MethodMap &, selector_ptr & );
  }
}

#endif
