/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.4 2008/10/08 22:07:25 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Type.h"
#include "boost/thread/tss.hpp"
#include "FWCore/Utilities/interface/UseReflex.h"

namespace edm {
  void
  TypeID::print(std::ostream& os) const {
    os << className();
  }

  static
  std::string typeToClassName(const std::type_info& iType) {
    Reflex::Type t = Reflex::Type::ByTypeInfo(iType);
    if (!bool(t)) {
      throw edm::Exception(errors::DictionaryNotFound,"NoMatch")
      << "TypeID::className: No dictionary for class " << iType.name() << '\n';
    }
    return t.Name(Reflex::SCOPED);
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
    std::string theName = className();
    if (theName.find("edm::Wrapper") == 0) {
      stripTemplate(theName);
    }
    return theName;
  }

  std::string 
  TypeID::friendlyClassName() const {
    return friendlyname::friendlyName(className());
  }

  bool
  TypeID::stripTemplate(std::string& theName) {
    std::string const spec("<,>");
    char const space = ' ';
    std::string::size_type idx = theName.find_first_of(spec);
    if (idx == std::string::npos) {
      return false;
    }
    std::string::size_type first = 0;
    std::string::size_type after = idx;
    if (theName[idx] == '<') {
      after = theName.rfind('>');
      assert (after != std::string::npos);
      first = ++idx;
    } else {
      theName = theName.substr(0, idx);
    }
    std::string::size_type idxa = after;
    while (space == theName[--idxa]) --after;
    std::string::size_type idxf = first;
    while (space == theName[idxf++]) ++first;
    theName = theName.substr(first, after - first);
    return true;
  }

  bool
  TypeID::stripNamespace(std::string& theName) {
    std::string::size_type idx = theName.rfind(':');
    bool ret = (idx != std::string::npos);
    if (ret) {
      ++idx;
      theName = theName.substr(idx);
    }
    return ret;
  }

  bool
  TypeID::hasDictionary() const {
    return bool(Reflex::Type::ByTypeInfo(typeInfo()));
  }

  std::ostream&
  operator<<(std::ostream& os, const TypeID& id) {
    id.print(os);
    return os;
  }
}

