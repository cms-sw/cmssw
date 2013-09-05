#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <ostream>

namespace edm {

  InternalContext::InternalContext(EventID const& eventID,
                                   ModuleCallingContext const* moduleCallingContext) :
    eventID_(eventID),
    moduleCallingContext_(moduleCallingContext) {
  }

  std::ostream& operator<<(std::ostream& os, InternalContext const& ic) {
    os << "InternalContext " << ic.eventID() << "\n";
    if(ic.moduleCallingContext()) {
      os << "    " << *ic.moduleCallingContext(); 
    }
    return os;
  }
}
