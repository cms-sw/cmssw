/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include <cassert>
#include <map>
#include <ostream>
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
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
}

  std::string const&
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

  std::string
  stripNamespace(std::string const& theName) {
    // Find last colon
    std::string::size_type colonIndex = theName.rfind(':');
    if(colonIndex == std::string::npos) {
      // No colons, so no namespace to strip
      return theName;
    }
    std::string::size_type bracketIndex = theName.rfind('>');
    if(bracketIndex == std::string::npos || bracketIndex < colonIndex) {
      // No '>' after last colon.  Strip up to and including last colon.
      return theName.substr(colonIndex+1);
    }
    // There is a '>' after the last colon.
    int depth = 1;
    for(size_t index = bracketIndex; index != 0; --index) {
      char c = theName[index - 1]; 
      if(c == '>') {
        ++depth;
      } else if(c == '<') {
        --depth;
        assert(depth >= 0);
      } else if(depth == 0 && c == ':') {
        return theName.substr(index);
      }
    }
    return theName;
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

