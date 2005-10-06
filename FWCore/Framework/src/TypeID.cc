/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.9 2005/09/30 22:05:33 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/Framework/src/TypeID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Type.h"
#include <string>

namespace edm {
  void
  TypeID::print(std::ostream& os) const {
    os << className();
  }

  std::string
  TypeID::className() const {
    seal::reflex::Type t = seal::reflex::Type::byTypeInfo(t_);
    if (!bool(t)) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "TypeID::className: No dictionary for class " << t_.name() << '\n';
    }
    return t.name(seal::reflex::SCOPED);
  }

  std::string 
  TypeID::userClassName() const {
    std::string name = className();
    if (name.find("edm::Wrapper") == 0) {
      stripTemplate(name);
    }
    return name;
  }

  std::string 
  TypeID::friendlyClassName() const {
    std::string name = userClassName();
    while (stripTemplate(name)) {}
    stripNamespace(name);
    return name;
  }

  bool
  TypeID::stripTemplate(std::string& name) {
    std::string::size_type idx = name.find('<');
    bool ret = (idx != std::string::npos);
    if (ret) {
      std::string::size_type idx2 = name.rfind('>');
      assert (idx2 != std::string::npos);
      std::string::size_type idx3 = idx2;
      char const space(' ');
      while (space == name[--idx3]) --idx2;
      ++idx;
      name = name.substr(idx, idx2 - idx);
    }
    return ret;
  }

  bool
  TypeID::stripNamespace(std::string& name) {
    std::string::size_type idx = name.rfind(':');
    bool ret = (idx != std::string::npos);
    if (ret) {
      ++idx;
      name = name.substr(idx);
    }
    return ret;
  }

  std::ostream&
  operator<<(std::ostream& os, const TypeID& id) {
    id.print(os);
    return os;
  }
}

