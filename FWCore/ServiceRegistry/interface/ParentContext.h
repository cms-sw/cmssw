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
    enum class Type { kGlobal, kInternal, kModule, kPlaceInPath, kStream, kInvalid };

    ParentContext() noexcept;
    ParentContext(GlobalContext const*) noexcept;
    ParentContext(InternalContext const*) noexcept;
    ParentContext(ModuleCallingContext const*) noexcept;
    ParentContext(PlaceInPathContext const*) noexcept;
    ParentContext(StreamContext const*) noexcept;

    [[nodiscard]] Type type() const noexcept { return type_; }

    bool isAtEndTransition() const noexcept;

    GlobalContext const* globalContext() const noexcept(false);
    InternalContext const* internalContext() const noexcept(false);
    ModuleCallingContext const* moduleCallingContext() const noexcept(false);
    PlaceInPathContext const* placeInPathContext() const noexcept(false);
    StreamContext const* streamContext() const noexcept(false);

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
}  // namespace edm
#endif
