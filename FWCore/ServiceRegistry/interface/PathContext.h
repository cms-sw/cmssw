#ifndef FWCore_ServiceRegistry_PathContext_h
#define FWCore_ServiceRegistry_PathContext_h

/**\class edm::PathContext

 Description: This is intended primarily to be passed to
Services as an argument to their callback functions.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/10/2013

#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <iosfwd>
#include <string>

namespace edm {

  class PathContext {
  public:

    PathContext(std::string const& pathName,
                unsigned int pathID,
                StreamContext const* streamContext);

    std::string const& pathName() const { return pathName_; }
    unsigned int pathID() const { return pathID_; }
    StreamContext const* streamContext() const { return streamContext_; }

  private:
    std::string pathName_;
    unsigned int pathID_;
    StreamContext const* streamContext_;
  };

  std::ostream& operator<<(std::ostream&, PathContext const&);
}
#endif
