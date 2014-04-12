#ifndef FWCore_ServiceRegistry_ParentContext_h
#define FWCore_ServiceRegistry_ParentContext_h

/**\class edm::ParentContext

 Description: This is intended to be used as a member
of ModuleCallingContext.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/11/2013

#include <iosfwd>

namespace edm {

  class GlobalContext;
  class InternalContext;
  class ModuleCallingContext;
  class PlaceInPathContext;
  class StreamContext;

  class ParentContext {
  public:

    enum class Type {
      kGlobal,
      kInternal,
      kModule,
      kPlaceInPath,
      kStream,
      kInvalid
    };

    ParentContext();
    ParentContext(GlobalContext const*);
    ParentContext(InternalContext const*);
    ParentContext(ModuleCallingContext const*);
    ParentContext(PlaceInPathContext const*);
    ParentContext(StreamContext const*);

    Type type() const { return type_; }

    GlobalContext const* globalContext() const;
    InternalContext const* internalContext() const;
    ModuleCallingContext const* moduleCallingContext() const;
    PlaceInPathContext const* placeInPathContext() const;
    StreamContext const* streamContext() const;

  private:
    Type type_;

    union Parent {
      GlobalContext const* global;
      InternalContext const* internal;
      ModuleCallingContext const* module;
      PlaceInPathContext const* placeInPath;
      StreamContext const* stream;
    } parent_;
  };

  std::ostream& operator<<(std::ostream&, ParentContext const&);
}
#endif
