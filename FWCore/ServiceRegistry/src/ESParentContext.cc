#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ESParentContext::ESParentContext() : type_(Type::kInvalid) { parent_.esmodule = nullptr; }

  ESParentContext::ESParentContext(ModuleCallingContext const* module) noexcept : type_(Type::kModule) {
    parent_.module = module;
  }

  ESParentContext::ESParentContext(ESModuleCallingContext const* module) noexcept : type_(Type::kESModule) {
    parent_.esmodule = module;
  }

  ModuleCallingContext const* ESParentContext::moduleCallingContext() const {
    if (type_ != Type::kModule) {
      throw Exception(errors::LogicError)
          << "ESParentContext::moduleCallingContext called for incorrect type of context";
    }
    return parent_.module;
  }

  ESModuleCallingContext const* ESParentContext::esmoduleCallingContext() const {
    if (type_ != Type::kESModule) {
      throw Exception(errors::LogicError)
          << "ESParentContext::esmoduleCallingContext called for incorrect type of context";
    }
    return parent_.esmodule;
  }
}  // namespace edm
