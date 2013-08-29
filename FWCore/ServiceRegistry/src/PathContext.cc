#include "FWCore/ServiceRegistry/interface/PathContext.h"

#include <ostream>

namespace edm {

  PathContext::PathContext(std::string const& pathName,
                           unsigned int pathID,
                           StreamContext const* streamContext) :
    pathName_(pathName),
    pathID_(pathID),
    streamContext_(streamContext) {
  }

  std::ostream& operator<<(std::ostream& os, PathContext const& pc) {
    os << "PathContext: pathName = " << pc.pathName()
       << " pathID = " << pc.pathID() << "\n";
    if(pc.streamContext()) {
      os << "    " << *pc.streamContext(); 
    }
    return os;
  }
}
