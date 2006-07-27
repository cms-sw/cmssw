#if !defined(PACKAGE_CUTPARSER_H)
#define PACKAGE_CUTPARSER_H
#include "PhysicsTools/UtilAlgos/interface/ReflexSelector.h"
#include "PhysicsTools/UtilAlgos/interface/methods.h"
#include <string>
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    typedef boost::shared_ptr<ReflexSelector> selector_ptr;
    bool cutParser( const std::string & ,
		    const methods::methodMap & ,
		    selector_ptr &  );
  }
}

#endif
