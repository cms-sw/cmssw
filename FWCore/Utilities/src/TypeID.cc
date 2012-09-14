/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include <cassert>
#include <ostream>
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "Reflex/Type.h"
#include "boost/thread/tss.hpp"

namespace edm {
  void
  TypeID::print(std::ostream& os) const {
    try {
      os << className();
    } catch (cms::Exception const& e) {
      os << typeInfo().name();
    }
  }

namespace {

  TypeID const nullTypeID;

  std::string typeToClassName(std::type_info const& iType) {
    Reflex::Type t = Reflex::Type::ByTypeInfo(iType);
    if (!bool(t)) {
      std::string result;
      try {
        typeDemangle(iType.name(), result);
      } catch (cms::Exception const& e) {
        cms::Exception theError("Name Demangling Error");
        theError << "TypeID::typeToClassName: can't demangle " << iType.name() << '\n';
        theError.append(e);
        throw theError;
      }
      return result;
    }
    return t.Name(Reflex::SCOPED | Reflex::FINAL);
  }

  std::type_info const* classNameToType(std::string const& className) {
    Reflex::Type t = Reflex::Type::ByName(className);
    if (!bool(t)) {
      return 0;
    }
    return &t.TypeInfo();
  }
}

  TypeID
  TypeID::byName(std::string const& className) {
    std::type_info const* t = classNameToType(className);
    return(t != 0 ? TypeID(*t) : TypeID());
  }

  std::string
  TypeID::className() const {
    typedef std::map<edm::TypeID, std::string> Map;
    static boost::thread_specific_ptr<Map> s_typeToName;
    if(0 == s_typeToName.get()){
      s_typeToName.reset(new Map);
    }
    Map::const_iterator itFound = s_typeToName->find(*this);
    if(s_typeToName->end() == itFound) {
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
  stripTemplate(std::string& theName) {
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
  stripNamespace(std::string& theName) {
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

  bool
  TypeID::isComplete() const {
    Reflex::Type t = Reflex::Type::ByTypeInfo(typeInfo());
    return bool(t) && t.IsComplete();
  }

  TypeID::operator bool() const {
    return !(*this == nullTypeID);
  }


  std::ostream&
  operator<<(std::ostream& os, TypeID const& id) {
    id.print(os);
    return os;
  }
}

