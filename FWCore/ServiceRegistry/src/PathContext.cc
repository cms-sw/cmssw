#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <ostream>

namespace edm {

  PathContext::PathContext(std::string const& pathName,
                           StreamContext const* streamContext,
                           unsigned int pathID,
                           PathType pathType) :
    pathName_(pathName),
    streamContext_(streamContext),
    pathID_(pathID),
    pathType_(pathType) {
  }

  std::ostream& operator<<(std::ostream& os, PathContext const& pc) {
    os << "PathContext: pathName = " << pc.pathName()
       << " pathID = " << pc.pathID();
    if(pc.pathType() == PathContext::PathType::kEndPath) {
      os << " (EndPath)\n"; 
    } else {
      os << "\n";
    }
    if(pc.streamContext()) {
      os << "    " << *pc.streamContext(); 
    }
    return os;
  }
}
