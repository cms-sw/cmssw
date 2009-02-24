#include "CommonTools/Utils/src/returnType.h"
#include <map>
#include <string>
using namespace Reflex;
using namespace std;
using namespace reco::method;

namespace reco {
  Type returnType(const Member & mem) {
    Type t = mem.TypeOf().ReturnType();
    if(t) {
       while(t.IsTypedef()) t = t.ToType();
    }
    return t;
  }

  TypeCode returnTypeCode(const Member & mem) {
    return typeCode(returnType(mem));
  }

  TypeCode typeCode(const Type & t) {
    static map<string, method::TypeCode> retTypeMap;
    if (retTypeMap.size() == 0) {
      retTypeMap["double"] = doubleType;
      retTypeMap["float"] = floatType;
      retTypeMap["int"] = intType;
      retTypeMap["unsigned int"] = uIntType;
      retTypeMap["short"] = shortType;
      retTypeMap["unsigned short"] = uShortType;
      retTypeMap["long"] = longType;
      retTypeMap["unsigned long"] = uLongType;
      retTypeMap["size_t"] = uLongType;
      retTypeMap["char"] = charType;
      retTypeMap["unsigned char"] = uCharType;
      retTypeMap["bool"] = boolType;
    }
    map<string, TypeCode>::const_iterator f = retTypeMap.find(t.Name());
    if (f == retTypeMap.end()) return invalid;
    else return f->second;
  }
}
