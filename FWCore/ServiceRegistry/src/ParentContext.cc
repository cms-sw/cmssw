#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ParentContext::ParentContext() :
    type_(Type::kInvalid) {
    parent_.global = nullptr;
  }

  ParentContext::ParentContext(GlobalContext const* global) :
    type_(Type::kGlobal) {
    parent_.global = global;
  }

  ParentContext::ParentContext(InternalContext const* internal) :
    type_(Type::kInternal) {
    parent_.internal = internal;
  }

  ParentContext::ParentContext(ModuleCallingContext const* module) :
    type_(Type::kModule) {
    parent_.module = module;
  }

  ParentContext::ParentContext(PathContext const* path) :
    type_(Type::kPath) {
    parent_.path = path;
  }

  ParentContext::ParentContext(StreamContext const* stream) :
    type_(Type::kStream) {
    parent_.stream = stream;
  }

  ModuleCallingContext const*
  ParentContext::moduleCallingContext() const {
    if(type_ != Type::kModule) {
      throw Exception(errors::LogicError)
        << "ParentContext::moduleCallingContext called for incorrect type of context";
    }
    return parent_.module;
  }

  PathContext const*
  ParentContext::pathContext() const {
    if(type_ != Type::kPath) {
      throw Exception(errors::LogicError)
        << "ParentContext::pathContext called for incorrect type of context";
    }
    return parent_.path;
  }

  StreamContext const*
  ParentContext::streamContext() const {
    if(type_ != Type::kStream) {
      throw Exception(errors::LogicError)
        << "ParentContext::streamContext called for incorrect type of context";
    }
    return parent_.stream;
  }

  GlobalContext const*
  ParentContext::globalContext() const {
    if(type_ != Type::kGlobal) {
      throw Exception(errors::LogicError)
        << "ParentContext::globalContext called for incorrect type of context";
    }
    return parent_.global;
  }

  InternalContext const*
  ParentContext::internalContext() const {
    if(type_ != Type::kInternal) {
      throw Exception(errors::LogicError)
        << "ParentContext::internalContext called for incorrect type of context";
    }
    return parent_.internal;
  }

  std::ostream& operator<<(std::ostream& os, ParentContext const& pc) {
    if(pc.type() == ParentContext::Type::kGlobal && pc.globalContext()) {
      os << *pc.globalContext();
    } else if (pc.type() == ParentContext::Type::kInternal && pc.internalContext()) {
      os << *pc.internalContext();
    } else if (pc.type() == ParentContext::Type::kModule && pc.moduleCallingContext()) {
      os << *pc.moduleCallingContext();
    } else if (pc.type() == ParentContext::Type::kPath && pc.pathContext()) {
      os << *pc.pathContext();
    } else if (pc.type() == ParentContext::Type::kStream && pc.streamContext()) {
      os << *pc.streamContext();
    } else if (pc.type() == ParentContext::Type::kInvalid) {
      os << "ParentContext invalid\n";
    }
    return os;
  }
}
