#ifndef FWCore_ServiceRegistry_ESParentContext_h
#define FWCore_ServiceRegistry_ESParentContext_h

/**\class edm::ESParentContext

 Description: This is intended to be used as a member of ESModuleCallingContext.

 Usage:


*/
//
// Original Author: C. Jones
//         Created: 2/07/2021

namespace edm {

  class ModuleCallingContext;
  class ESModuleCallingContext;

  class ESParentContext {
  public:
    enum class Type { kModule, kESModule, kInvalid };

    ESParentContext();
    explicit ESParentContext(ModuleCallingContext const*) noexcept;
    explicit ESParentContext(ESModuleCallingContext const*) noexcept;

    Type type() const noexcept { return type_; }

    ModuleCallingContext const* moduleCallingContext() const;
    ESModuleCallingContext const* esmoduleCallingContext() const;

  private:
    Type type_;

    union Parent {
      ModuleCallingContext const* module;
      ESModuleCallingContext const* esmodule;
    } parent_;
  };
}  // namespace edm
#endif
