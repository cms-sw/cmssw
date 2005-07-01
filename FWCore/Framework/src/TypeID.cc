/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.2 2005/06/23 22:01:31 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/CoreFramework/src/TypeID.h"
#include "Reflection/Class.h"
#include <string>

namespace edm {
  void
  TypeID::print(std::ostream& os) const {
    os << t_.name();
  }

  std::string
  TypeID::reflectionClassName() const {
    seal::reflect::Class const * c = seal::reflect::Class::forTypeinfo(t_);
    if (c == 0) {
      throw edm::Exception(edm::errors::Configuration,"MissingType")
        << "No SEAL Reflection entry for class: " << t_.name();
    }
    return c->fullName();
  }

  std::string 
  TypeID::userClassName() const {
    char const space(' ');
    std::string name = reflectionClassName();
    if (name.find("edm::Wrapper") == 0) {
	std::string::size_type idx = name.find('<');
	std::string::size_type idx2 = name.rfind('>');
	assert (idx != std::string::npos);
	assert (idx2 != std::string::npos);
        std::string::size_type idx3 = idx2;
	while (space == name[--idx3]) --idx2;
	++idx;
	name = name.substr(idx, idx2 - idx);
    }
    return name;
  }

  std::ostream&
  operator<<(std::ostream& os, const TypeID& id) {
    id.print(os);
    return os;
  }
}

