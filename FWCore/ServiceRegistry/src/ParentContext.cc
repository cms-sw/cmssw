#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
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

  ParentContext::ParentContext(PlaceInPathContext const* placeInPath) :
    type_(Type::kPlaceInPath) {
    parent_.placeInPath = placeInPath;
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

  PlaceInPathContext const*
  ParentContext::placeInPathContext() const {
    if(type_ != Type::kPlaceInPath) {
      throw Exception(errors::LogicError)
        << "ParentContext::placeInPathContext called for incorrect type of context";
    }
    return parent_.placeInPath;
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
    } else if (pc.type() == ParentContext::Type::kPlaceInPath && pc.placeInPathContext()) {
      os << *pc.placeInPathContext();
    } else if (pc.type() == ParentContext::Type::kStream && pc.streamContext()) {
      os << *pc.streamContext();
    } else if (pc.type() == ParentContext::Type::kInvalid) {
      os << "ParentContext invalid\n";
    }
    return os;
  }
}
