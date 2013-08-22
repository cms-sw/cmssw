#ifndef FWCore_ServiceRegistry_PlaceInPathContext_h
#define FWCore_ServiceRegistry_PlaceInPathContext_h

/**\class edm::PlaceInPathContext

 Description: Holds context information to indentify
 the position within a sequence of modules in a path.
*/
//
// Original Author: W. David Dagenhart
//         Created: 7/31/2013

#include <iosfwd>

namespace edm {

  class PathContext;

  class PlaceInPathContext {

  public:

    PlaceInPathContext(unsigned int);

    unsigned int placeInPath() const { return placeInPath_; }
    PathContext const* pathContext() const { return pathContext_; }

    void setPathContext(PathContext const* v) { pathContext_ = v; }

  private:
    unsigned int placeInPath_;
    PathContext const* pathContext_;
  };

  std::ostream& operator<<(std::ostream&, PlaceInPathContext const&);
}
#endif
