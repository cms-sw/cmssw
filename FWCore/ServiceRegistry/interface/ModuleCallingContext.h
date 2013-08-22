#ifndef FWCore_ServiceRegistry_ModuleCallingContext_h
#define FWCore_ServiceRegistry_ModuleCallingContext_h

/**\class edm::ModuleCallingContext

 Description: This is intended primarily to be passed to
Services as an argument to their callback functions.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/11/2013

#include "FWCore/ServiceRegistry/interface/ParentContext.h"

#include <iosfwd>

namespace edm {

  class GlobalContext;
  class InternalContext;
  class ModuleDescription;
  class PlaceInPathContext;
  class StreamContext;


  class ModuleCallingContext {
  public:

    typedef ParentContext::Type Type;

    enum class State {
      kPrefetching,  // prefetching products before starting to run
      kRunning,      // module actually running
      kInvalid
    };

    ModuleCallingContext(ModuleDescription const* moduleDescription);
    ModuleCallingContext(ModuleDescription const* moduleDescription, State state, ParentContext const& parent);

    void setContext(State state, ParentContext const& parent);

    ModuleDescription const* moduleDescription() const { return moduleDescription_; }
    State state() const { return state_; }
    Type type() const { return parent_.type(); }
    ParentContext const& parent() const { return parent_; }
    ModuleCallingContext const* moduleCallingContext() const { return parent_.moduleCallingContext(); }
    PlaceInPathContext const* placeInPathContext() const { return parent_.placeInPathContext(); }
    StreamContext const* streamContext() const { return parent_.streamContext(); }
    GlobalContext const* globalContext() const { return parent_.globalContext(); }
    InternalContext const* internalContext() const { return parent_.internalContext(); }

    // These functions will iterate up a series of linked context objects
    // to find the StreamContext or GlobalContext at the top of the series.
    // Throws if the top context object does not have that type.
    StreamContext const* getStreamContext() const;
    GlobalContext const* getGlobalContext() const;

  private:

    ModuleDescription const* moduleDescription_;
    ParentContext parent_;
    State state_;
  };

  std::ostream& operator<<(std::ostream&, ModuleCallingContext const&);
}
#endif
