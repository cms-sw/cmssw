#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ESModuleCallingContext::ESModuleCallingContext(edm::eventsetup::ComponentDescription const* componentDescription)
      : componentDescription_(componentDescription), parent_(), state_(State::kInvalid) {}

  ESModuleCallingContext::ESModuleCallingContext(edm::eventsetup::ComponentDescription const* componentDescription,
                                                 State state,
                                                 ESParentContext const& parent)
      : componentDescription_(componentDescription), parent_(parent), state_(state) {}

  void ESModuleCallingContext::setContext(State state, ESParentContext const& parent) {
    state_ = state;
    parent_ = parent;
  }

  ModuleCallingContext const* ESModuleCallingContext::getTopModuleCallingContext() const {
    ESModuleCallingContext const* mcc = this;
    while (mcc->type() == ESParentContext::Type::kESModule) {
      mcc = mcc->esmoduleCallingContext();
    }
    return mcc->moduleCallingContext()->getTopModuleCallingContext();
  }

  unsigned ESModuleCallingContext::depth() const {
    unsigned depth = 0;
    ESModuleCallingContext const* mcc = this;
    while (mcc->type() == ESParentContext::Type::kESModule) {
      ++depth;
      mcc = mcc->esmoduleCallingContext();
    }
    return depth + mcc->moduleCallingContext()->depth();
  }

}  // namespace edm
