#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"

#include <ostream>

namespace edm {

  PlaceInPathContext::PlaceInPathContext(unsigned int placeInPath) :
    placeInPath_(placeInPath),
    pathContext_(nullptr) {
  }

  std::ostream& operator<<(std::ostream& os, PlaceInPathContext const& ppc) {
    os << "PlaceInPathContext " << ppc.placeInPath() << "\n";
    if(ppc.pathContext()) {
      os << "    " << *ppc.pathContext(); 
    }
    return os;
  }
}
