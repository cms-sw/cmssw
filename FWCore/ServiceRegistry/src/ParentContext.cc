#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ParentContext::ParentContext() noexcept : parent_(std::monostate{}) {}

  ParentContext::ParentContext(GlobalContext const* global) noexcept : parent_(global) {}

  ParentContext::ParentContext(InternalContext const* internal) noexcept : parent_(internal) {}

  ParentContext::ParentContext(ModuleCallingContext const* module) noexcept : parent_(module) {}

  ParentContext::ParentContext(PlaceInPathContext const* placeInPath) noexcept : parent_(placeInPath) {}

  ParentContext::ParentContext(StreamContext const* stream) noexcept : parent_(stream) {}

  ModuleCallingContext const* ParentContext::moduleCallingContext() const noexcept(false) {
    if (type() != Type::kModule) {
      throw Exception(errors::LogicError) << "ParentContext::moduleCallingContext called for incorrect type of context";
    }
    return std::get<ModuleCallingContext const*>(parent_);
  }

  PlaceInPathContext const* ParentContext::placeInPathContext() const noexcept(false) {
    if (type() != Type::kPlaceInPath) {
      throw Exception(errors::LogicError) << "ParentContext::placeInPathContext called for incorrect type of context";
    }
    return std::get<PlaceInPathContext const*>(parent_);
  }

  StreamContext const* ParentContext::streamContext() const noexcept(false) {
    if (type() != Type::kStream) {
      throw Exception(errors::LogicError) << "ParentContext::streamContext called for incorrect type of context";
    }
    return std::get<StreamContext const*>(parent_);
  }

  GlobalContext const* ParentContext::globalContext() const noexcept(false) {
    if (type() != Type::kGlobal) {
      throw Exception(errors::LogicError) << "ParentContext::globalContext called for incorrect type of context";
    }
    return std::get<GlobalContext const*>(parent_);
  }

  InternalContext const* ParentContext::internalContext() const noexcept(false) {
    if (type() != Type::kInternal) {
      throw Exception(errors::LogicError) << "ParentContext::internalContext called for incorrect type of context";
    }
    return std::get<InternalContext const*>(parent_);
  }

  bool ParentContext::isAtEndTransition() const noexcept {
    switch (type()) {
      case Type::kGlobal: {
        return std::get<GlobalContext const*>(parent_)->isAtEndTransition();
      }
      case Type::kModule: {
        return std::get<ModuleCallingContext const*>(parent_)->parent().isAtEndTransition();
      }
      case Type::kStream: {
        return std::get<StreamContext const*>(parent_)->isAtEndTransition();
      }
      case Type::kPlaceInPath: {
        return std::get<PlaceInPathContext const*>(parent_)->pathContext()->streamContext()->isAtEndTransition();
      }
      default:
        break;
    }
    return false;
  }

  std::ostream& operator<<(std::ostream& os, ParentContext const& pc) {
    if (pc.type() == ParentContext::Type::kGlobal && pc.globalContext()) {
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
}  // namespace edm
