#include "CommonTools/Utils/src/returnType.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace reco::method;
using namespace std;

namespace reco {

  edm::TypeWithDict returnType(const edm::FunctionWithDict& func) {
    return func.finalReturnType();
  }

  TypeCode returnTypeCode(const edm::FunctionWithDict& func) {
    return typeCode(returnType(func));
  }

  //this is already alphabetized
  static const std::vector<std::pair<char const* const, method::TypeCode> > retTypeVec {
     {"bool", boolType},
     {"char", charType},
     {"double", doubleType},
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
     {"unsigned long int", uLongType},
     {"unsigned short", uShortType},
     {"unsigned short int", uShortType}
  };

  TypeCode typeCode(const edm::TypeWithDict& t) {
    typedef std::pair<const char* const, method::TypeCode> Values;
    std::string name = t.name();
    auto f = std::equal_range(retTypeVec.begin(), retTypeVec.end(),
      Values{name.c_str(), enumType},
      [](const Values& iLHS, const Values& iRHS) -> bool {
        return std::strcmp(iLHS.first, iRHS.first) < 0;
      });
    if (f.first == f.second) {
      return t.isEnum() ? enumType : invalid;
    }
    return f.first->second;
  }

} // namespace reco

