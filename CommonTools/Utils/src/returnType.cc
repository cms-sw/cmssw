#include "CommonTools/Utils/src/returnType.h"
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
using namespace std;
using namespace reco::method;

namespace reco {
  edm::TypeWithDict returnType(const edm::FunctionWithDict & mem) {
    return mem.finalReturnType();
  }

  TypeCode returnTypeCode(const edm::FunctionWithDict & mem) {
    return typeCode(returnType(mem));
  }

  //this is already alphabetized
  static const std::vector<std::pair<char const * const, method::TypeCode> > retTypeVec {
     {"bool",boolType},
     {"char",charType},
     {"double",doubleType},
     {"float", floatType},
     {"int", intType},
     {"long", longType},
     {"long int", longType},
     {"short", shortType},
     {"short int", shortType},
     {"size_t", uLongType},
     {"unsigned char", uCharType},
     {"unsigned int", uIntType},
     {"unsigned long", uLongType},
     {"unsigned long int",uLongType},
     {"unsigned short",uShortType},
     {"unsigned short int",uShortType}
  };

  TypeCode typeCode(const edm::TypeWithDict & t) {

    typedef std::pair<const char*const, method::TypeCode> Values;
    std::string const theName(t.name());
    char const* name = theName.c_str();
    auto f = std::equal_range(retTypeVec.begin(),retTypeVec.end(),Values{name,enumType},
                              [] ( Values const& iLHS, Values const& iRHS) -> bool{
       return std::strcmp(iLHS.first,iRHS.first)<0;
    });
    if (f.first == f.second) return (t.isEnum() ? enumType : invalid);
    else return f.first->second;
  }
}
