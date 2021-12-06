#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <ostream>

namespace edm {

  ModuleCallingContext::ModuleCallingContext(ModuleDescription const* moduleDescription)
      : previousModuleOnThread_(nullptr), moduleDescription_(moduleDescription), parent_(), state_(State::kInvalid) {}

  ModuleCallingContext::ModuleCallingContext(ModuleDescription const* moduleDescription,
                                             State state,
                                             ParentContext const& parent,
                                             ModuleCallingContext const* previousOnThread)
      : previousModuleOnThread_(previousOnThread),
        moduleDescription_(moduleDescription),
        parent_(parent),
        state_(state) {}

  void ModuleCallingContext::setContext(State state,
                                        ParentContext const& parent,
                                        ModuleCallingContext const* previousOnThread) {
    state_ = state;
    parent_ = parent;
    previousModuleOnThread_ = previousOnThread;
  }

  StreamContext const* ModuleCallingContext::getStreamContext() const {
    ModuleCallingContext const* mcc = getTopModuleCallingContext();
    if (mcc->type() == ParentContext::Type::kPlaceInPath) {
      return mcc->placeInPathContext()->pathContext()->streamContext();
    } else if (mcc->type() != ParentContext::Type::kStream) {
      throw Exception(errors::LogicError)
          << "ModuleCallingContext::getStreamContext() called in context not linked to a StreamContext\n";
    }
    return mcc->streamContext();
  }

  GlobalContext const* ModuleCallingContext::getGlobalContext() const {
    ModuleCallingContext const* mcc = getTopModuleCallingContext();
    if (mcc->type() != ParentContext::Type::kGlobal) {
      throw Exception(errors::LogicError)
          << "ModuleCallingContext::getGlobalContext() called in context not linked to a GlobalContext\n";
    }
    return mcc->globalContext();
  }

  ModuleCallingContext const* ModuleCallingContext::getTopModuleCallingContext() const {
    ModuleCallingContext const* mcc = this;
    while (mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    if (mcc->type() == ParentContext::Type::kInternal) {
      mcc = mcc->internalContext()->moduleCallingContext();
    }
    while (mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    return mcc;
  }

  unsigned ModuleCallingContext::depth() const {
    unsigned depth = 0;
    ModuleCallingContext const* mcc = this;
    while (mcc->type() == ParentContext::Type::kModule) {
      ++depth;
      mcc = mcc->moduleCallingContext();
    }
    if (mcc->type() == ParentContext::Type::kInternal) {
      ++depth;
      mcc = mcc->internalContext()->moduleCallingContext();
    }
    while (mcc->type() == ParentContext::Type::kModule) {
      ++depth;
      mcc = mcc->moduleCallingContext();
    }
    return depth;
  }

  void exceptionContext(cms::Exception& ex, ModuleCallingContext const& mcc) {
    ModuleCallingContext const* imcc = &mcc;
    while ((imcc->type() == ParentContext::Type::kModule) or (imcc->type() == ParentContext::Type::kInternal)) {
      std::ostringstream iost;
      if (imcc->state() == ModuleCallingContext::State::kPrefetching) {
        iost << "Prefetching for module ";
      } else {
        iost << "Calling method for module ";
      }
      iost << imcc->moduleDescription()->moduleName() << "/'" << imcc->moduleDescription()->moduleLabel() << "'";

      if (imcc->type() == ParentContext::Type::kInternal) {
        iost << " (probably inside some kind of mixing module)";
        imcc = imcc->internalContext()->moduleCallingContext();
      } else {
        imcc = imcc->moduleCallingContext();
      }
      ex.addContext(iost.str());
    }
    std::ostringstream ost;
    if (imcc->state() == ModuleCallingContext::State::kPrefetching) {
      ost << "Prefetching for module ";
    } else {
      ost << "Calling method for module ";
    }
    ost << imcc->moduleDescription()->moduleName() << "/'" << imcc->moduleDescription()->moduleLabel() << "'";
    ex.addContext(ost.str());

    if (imcc->type() == ParentContext::Type::kPlaceInPath) {
      ost.str("");
      ost << "Running path '";
      ost << imcc->placeInPathContext()->pathContext()->pathName() << "'";
      ex.addContext(ost.str());
      auto streamContext = imcc->placeInPathContext()->pathContext()->streamContext();
      if (streamContext) {
        ost.str("");
        edm::exceptionContext(ost, *streamContext);
        ex.addContext(ost.str());
      }
    } else {
      if (imcc->type() == ParentContext::Type::kStream) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->streamContext()));
        ex.addContext(ost.str());
      } else if (imcc->type() == ParentContext::Type::kGlobal) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->globalContext()));
        ex.addContext(ost.str());
      }
    }
  }

  std::ostream& operator<<(std::ostream& os, ModuleCallingContext const& mcc) {
    os << "ModuleCallingContext state = ";
    switch (mcc.state()) {
      case ModuleCallingContext::State::kInvalid:
        os << "Invalid";
        break;
      case ModuleCallingContext::State::kPrefetching:
        os << "Prefetching";
        break;
      case ModuleCallingContext::State::kRunning:
        os << "Running";
        break;
    }
    os << "\n";
    if (mcc.state() == ModuleCallingContext::State::kInvalid) {
      return os;
    }
    if (mcc.moduleDescription()) {
      os << "    moduleDescription: " << *mcc.moduleDescription() << "\n";
    }
    os << "    " << mcc.parent();
    if (mcc.previousModuleOnThread()) {
      if (mcc.type() == ParentContext::Type::kModule && mcc.moduleCallingContext() == mcc.previousModuleOnThread()) {
        os << "    previousModuleOnThread: same as parent module\n";
      } else {
        os << "    previousModuleOnThread: " << *mcc.previousModuleOnThread();
      }
    }
    return os;
  }
}  // namespace edm
