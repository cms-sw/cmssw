#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <ostream>

namespace edm {

  ModuleCallingContext::ModuleCallingContext(ModuleDescription const* moduleDescription) :
    moduleDescription_(moduleDescription),
    parent_(),
    state_(State::kInvalid) {
  }

  ModuleCallingContext::ModuleCallingContext(ModuleDescription const* moduleDescription, State state, ParentContext const& parent) :
    moduleDescription_(moduleDescription),
    parent_(parent),
    state_(state) {
  }

  void ModuleCallingContext::setContext(State state, ParentContext const& parent) {
    state_ = state;
    parent_ = parent;
  }

  StreamContext const*
  ModuleCallingContext::getStreamContext() const {
    ModuleCallingContext const* mcc = this;
    while(mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    if(mcc->type() == ParentContext::Type::kInternal) {
      mcc = mcc->internalContext()->moduleCallingContext();
    }
    while(mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    if(mcc->type() == ParentContext::Type::kPath) {
      return mcc->pathContext()->streamContext();
    } else if (mcc->type() != ParentContext::Type::kStream) {
      throw Exception(errors::LogicError)
        << "ModuleCallingContext::getStreamContext() called in context not linked to a StreamContext\n";
    }
    return mcc->streamContext();
  }

  GlobalContext const*
  ModuleCallingContext::getGlobalContext() const {
    ModuleCallingContext const* mcc = this;
    while(mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    if(mcc->type() == ParentContext::Type::kInternal) {
      mcc = mcc->internalContext()->moduleCallingContext();
    }
    while(mcc->type() == ParentContext::Type::kModule) {
      mcc = mcc->moduleCallingContext();
    }
    if (mcc->type() != ParentContext::Type::kGlobal) {
      throw Exception(errors::LogicError)
        << "ModuleCallingContext::getGlobalContext() called in context not linked to a GlobalContext\n";
    }
    return mcc->globalContext();
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
    if(mcc.state() == ModuleCallingContext::State::kInvalid) {
      return os;
    }
    if(mcc.moduleDescription()) {
      os << "    moduleDescription: " << *mcc.moduleDescription() << "\n";
    }
    os << "    " << mcc.parent();
    return os;
  }
}
