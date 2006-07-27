/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.15 2006/03/10 21:14:21 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/Framework/interface/TypeID.h"
#include "FWCore/Framework/src/FriendlyName.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Type.h"
#include <string>
#include "boost/thread/tss.hpp"

namespace edm {
  void
  TypeID::print(std::ostream& os) const {
    os << className();
  }

  static 
  std::string typeToClassName(const std::type_info& iType) {
    ROOT::Reflex::Type t = ROOT::Reflex::Type::ByTypeInfo(iType);
    if (!bool(t)) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
      << "TypeID::className: No dictionary for class " << iType.name() << '\n';
    }
    return t.Name(ROOT::Reflex::SCOPED);
  }
  
  std::string
  TypeID::className() const {
    typedef std::map<edm::TypeIDBase, std::string> Map;
    static boost::thread_specific_ptr<Map> s_typeToName;
    if(0 == s_typeToName.get()){
      s_typeToName.reset(new Map);
    }
    Map::const_iterator itFound = s_typeToName->find(*this);
    if(s_typeToName->end()==itFound) {
      itFound = s_typeToName->insert(Map::value_type(*this, typeToClassName(typeInfo()))).first;
    }
    return itFound->second;
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
    return friendlyname::friendlyName(className());
  }

  bool
  TypeID::stripTemplate(std::string& name) {
    std::string const spec("<,>");
    char const space = ' ';
    std::string::size_type idx = name.find_first_of(spec);
    if (idx == std::string::npos) {
      return false;
    }
    std::string::size_type first = 0;
    std::string::size_type after = idx;
    if (name[idx] == '<') {
      after = name.rfind('>');
      assert (after != std::string::npos);
      first = ++idx;
    } else {
      name = name.substr(0, idx);
    }
    std::string::size_type idxa = after;
    while (space == name[--idxa]) --after;
    std::string::size_type idxf = first;
    while (space == name[idxf++]) ++first;
    name = name.substr(first, after - first);
    return true;
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

