#ifndef FWCore_ServiceRegistry_ESParentContext_h
#define FWCore_ServiceRegistry_ESParentContext_h

/**\class edm::ESParentContext

 Description: This is intended to be used as a member of ESModuleCallingContext.

 Usage:


*/
//
// Original Author: C. Jones
//         Created: 2/07/2021

#include <variant>

namespace edm {

  class ModuleCallingContext;
  class ESModuleCallingContext;

  class ESParentContext {
  public:
    enum class Type { kModule, kESModule, kInvalid };

    ESParentContext();
    explicit ESParentContext(ModuleCallingContext const*) noexcept;
    explicit ESParentContext(ESModuleCallingContext const*) noexcept;

    Type type() const noexcept { return static_cast<Type>(parent_.index()); }

    ModuleCallingContext const* moduleCallingContext() const;
    ESModuleCallingContext const* esmoduleCallingContext() const;

  private:
    std::variant<ModuleCallingContext const*, ESModuleCallingContext const*, std::monostate> parent_;
  };
}  // namespace edm
#endif
