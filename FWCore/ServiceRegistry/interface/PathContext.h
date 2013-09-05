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

#include <iosfwd>
#include <string>

namespace edm {

  class StreamContext;

  class PathContext {
  public:

    enum class PathType {
      kPath,
      kEndPath
    };

    PathContext(std::string const& pathName,
                StreamContext const* streamContext,
                unsigned int pathID,
                PathType pathType);

    std::string const& pathName() const { return pathName_; }
    StreamContext const* streamContext() const { return streamContext_; }
    unsigned int pathID() const { return pathID_; }
    PathType pathType() const { return pathType_; }

    bool isEndPath() const { return pathType_ == PathType::kEndPath; }

  private:
    std::string pathName_;
    StreamContext const* streamContext_;
    unsigned int pathID_;
    PathType pathType_;
  };

  std::ostream& operator<<(std::ostream&, PathContext const&);
}
#endif
