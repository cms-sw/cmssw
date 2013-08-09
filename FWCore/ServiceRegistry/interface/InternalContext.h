#ifndef FWCore_ServiceRegistry_InternalContext_h
#define FWCore_ServiceRegistry_InternalContext_h

/**\class edm::InternalContext

 Description: Holds context information for the link
 between a MixingModule context and a module called
 by unscheduled production adding data to a secondary
 Principal.
*/
//
// Original Author: W. David Dagenhart
//         Created: 7/31/2013

#include "DataFormats/Provenance/interface/EventID.h"

#include <iosfwd>

namespace edm {

  class ModuleCallingContext;

  class InternalContext {

  public:

    InternalContext(EventID const& eventID,
                    ModuleCallingContext const*);

    EventID const& eventID() const { return eventID_; } // event#==0 is a lumi, event#==0&lumi#==0 is a run
    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

  private:
    EventID eventID_; // event#==0 is a lumi, event#==0&lumi#==0 is a run
    ModuleCallingContext const* moduleCallingContext_;
  };

  std::ostream& operator<<(std::ostream&, InternalContext const&);
}
#endif
