#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ESParentContext::ESParentContext() : parent_(std::monostate{}) {}

  ESParentContext::ESParentContext(ModuleCallingContext const* module) noexcept : parent_(module) {}

  ESParentContext::ESParentContext(ESModuleCallingContext const* module) noexcept : parent_(module) {}

  ModuleCallingContext const* ESParentContext::moduleCallingContext() const {
    if (type() != Type::kModule) {
      throw Exception(errors::LogicError)
          << "ESParentContext::moduleCallingContext called for incorrect type of context";
    }
    return std::get<ModuleCallingContext const*>(parent_);
  }

  ESModuleCallingContext const* ESParentContext::esmoduleCallingContext() const {
    if (type() != Type::kESModule) {
      throw Exception(errors::LogicError)
          << "ESParentContext::esmoduleCallingContext called for incorrect type of context";
    }
    return std::get<ESModuleCallingContext const*>(parent_);
  }
}  // namespace edm
