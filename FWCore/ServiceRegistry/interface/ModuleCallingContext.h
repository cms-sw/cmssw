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
#include <cstdint>

namespace cms {
  class Exception;
}
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

    ModuleCallingContext(ModuleDescription const* moduleDescription,
                         std::uintptr_t id,
                         State state,
                         ParentContext const& parent,
                         ModuleCallingContext const* previousOnThread);

    void setContext(State state, ParentContext const& parent, ModuleCallingContext const* previousOnThread);

    void setState(State state) { state_ = state; }

    ModuleDescription const* moduleDescription() const { return moduleDescription_; }
    State state() const { return state_; }
    Type type() const { return parent_.type(); }
    /** Returns a unique id for this module to differentiate possibly concurrent calls to the module.
        The value returned may be large so not appropriate for an index lookup.
        A value of 0 denotes a call to produce, analyze or filter. Other values denote a transform.
    */
    std::uintptr_t callID() const { return id_; }
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

    // This function will iterate up a series of linked context objects to
    // find the highest level ModuleCallingContext. It will often return a
    // pointer to itself.
    ModuleCallingContext const* getTopModuleCallingContext() const;

    // Returns the number of ModuleCallingContexts above this ModuleCallingContext
    // in the series of linked context objects.
    unsigned depth() const;

    ModuleCallingContext const* previousModuleOnThread() const { return previousModuleOnThread_; }

  private:
    ModuleCallingContext const* previousModuleOnThread_;
    ModuleDescription const* moduleDescription_;
    ParentContext parent_;
    std::uintptr_t id_;
    State state_;
  };

  void exceptionContext(cms::Exception&, ModuleCallingContext const&);
  std::ostream& operator<<(std::ostream&, ModuleCallingContext const&);
}  // namespace edm
#endif
