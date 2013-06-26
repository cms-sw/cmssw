#include "CommonTools/Utils/src/returnType.h"
#include <map>
#include <string>
using namespace std;
using namespace reco::method;

namespace reco {
  edm::TypeWithDict returnType(const edm::FunctionWithDict & mem) {
    return mem.finalReturnType();
  }

  TypeCode returnTypeCode(const edm::FunctionWithDict & mem) {
    return typeCode(returnType(mem));
  }

  TypeCode typeCode(const edm::TypeWithDict & t) {
    static map<string, method::TypeCode> retTypeMap;
    if (retTypeMap.size() == 0) {
      retTypeMap["double"] = doubleType;
      retTypeMap["float"] = floatType;
      retTypeMap["int"] = intType;
      retTypeMap["unsigned int"] = uIntType;
      retTypeMap["short"] = shortType;
      retTypeMap["short int"] = shortType;
      retTypeMap["unsigned short"] = uShortType;
      retTypeMap["unsigned short int"] = uShortType;
      retTypeMap["long"] = longType;
      retTypeMap["long int"] = longType;
      retTypeMap["unsigned long"] = uLongType;
      retTypeMap["unsigned long int"] = uLongType;
      retTypeMap["size_t"] = uLongType;
      retTypeMap["char"] = charType;
      retTypeMap["unsigned char"] = uCharType;
      retTypeMap["bool"] = boolType;
    }
    map<string, TypeCode>::const_iterator f = retTypeMap.find(t.name());
    if (f == retTypeMap.end()) return (t.isEnum() ? enumType : invalid);
    else return f->second;
  }
}
