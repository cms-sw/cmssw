#ifndef FWCore_ServiceRegistry_ESModuleCallingContext_h
#define FWCore_ServiceRegistry_ESModuleCallingContext_h

/**\class edm::ESModuleCallingContext

 Description: This is intended primarily to be passed to
Services as an argument to their callback functions.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/11/2013

#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include <iosfwd>
#include <cstdint>

namespace edm {

  namespace eventsetup {
    struct ComponentDescription;
  }
  class ModuleCallingContext;
  class ESModuleCallingContext {
  public:
    using Type = ESParentContext::Type;

    enum class State {
      kPrefetching,  // prefetching products before starting to run
      kRunning,      // module actually running
      kInvalid
    };

    ESModuleCallingContext(edm::eventsetup::ComponentDescription const* moduleDescription, std::uintptr_t id);

    ESModuleCallingContext(edm::eventsetup::ComponentDescription const* moduleDescription,
                           std::uintptr_t id,
                           State state,
                           ESParentContext const& parent);

    void setContext(State state, ESParentContext const& parent);

    void setState(State state) { state_ = state; }

    edm::eventsetup::ComponentDescription const* componentDescription() const { return componentDescription_; }
    State state() const { return state_; }
    Type type() const { return parent_.type(); }
    /** Returns a unique id for this module to differentiate possibly concurrent calls to the module.
	    The value returned may be large so not appropriate for an index lookup.
	*/
    std::uintptr_t callID() const { return id_; }
    ESParentContext const& parent() const { return parent_; }
    ModuleCallingContext const* moduleCallingContext() const { return parent_.moduleCallingContext(); }
    ESModuleCallingContext const* esmoduleCallingContext() const { return parent_.esmoduleCallingContext(); }

    // This function will iterate up a series of linked context objects to
    // find the highest level ModuleCallingContext.
    ModuleCallingContext const* getTopModuleCallingContext() const;

    // Returns the number of ESModuleCallingContexts above this ESModuleCallingContext
    // in the series of linked context objects.
    unsigned depth() const;

  private:
    edm::eventsetup::ComponentDescription const* componentDescription_;
    ESParentContext parent_;
    std::uintptr_t id_;
    State state_;
  };
}  // namespace edm
#endif
