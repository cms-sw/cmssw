/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"
#include "FWCore/Utilities/interface/EDMException.h"
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
#if GCC_PREREQUISITE(3,0,0)
      // demangling supported for currently supported gcc compilers.
      try {
        std::string result;
        typeDemangle(iType.name(), result);
        return result;
      } catch (cms::Exception const& e) {
        edm::Exception theError(errors::DictionaryNotFound,"NoMatch");
        theError << "TypeID::typeToClassName: No dictionary for class " << iType.name() << '\n';
        theError.append(e);
        throw theError;
      }
#else
      throw edm::Exception(errors::DictionaryNotFound,"NoMatch")
       << "TypeID::className: No dictionary for class " << iType.name() << '\n';
#endif
    }
    return t.Name(Reflex::SCOPED);
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

