#ifndef FWCore_Utilities_FunctionWithDict_h
#define FWCore_Utilities_FunctionWithDict_h

/*----------------------------------------------------------------------

FunctionWithDict:  A holder for a class member function

----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/IterWithDict.h"

#include "TMethod.h"
#include "TMethodArg.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace edm {

class ObjectWithDict;
class TypeWithDict;

class FunctionWithDict {
private:
  TMethod* function_;
public:
  FunctionWithDict();
  explicit FunctionWithDict(TMethod*);
  explicit operator bool() const;
  std::string name() const;
  std::string typeName() const;
  TypeWithDict typeOf() const;
  TypeWithDict returnType() const;
  TypeWithDict finalReturnType() const;
  bool isConst() const;
  bool isConstructor() const;
  bool isDestructor() const;
  bool isOperator() const;
  bool isPublic() const;
  bool isStatic() const;
  TypeWithDict declaringType() const;
  size_t functionParameterSize(bool required = false) const;
  size_t size() const;
  void invoke(ObjectWithDict const& obj, ObjectWithDict* ret = nullptr, std::vector<void*> const& values = std::vector<void*>()) const;
  void invoke(ObjectWithDict* ret = nullptr, std::vector<void*> const& values = std::vector<void*>()) const;
  IterWithDict<TMethodArg> begin() const;
  IterWithDict<TMethodArg> end() const;
};

} // namespace edm

#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

/// Call a static function of class theType, with a return
/// value of type T, by name with no arguments.
template<typename T>
inline
void
invokeByName(T& retval, TypeWithDict const& theType, std::string const& name)
{
  if (!bool(theType)) {
    fprintf(stderr, "FunctionWithDict: invokeByName<%s>: "
            "Passed type is invalid!\n", typeid(T).name());
    abort();
  }
  FunctionWithDict func = theType.functionMemberByName(name);
  if (!bool(func)) {
    fprintf(stderr, "FunctionWithDict: invokeByName<%s>: "
            "Could not find function named '%s' in type '%s'\n",
            typeid(T).name(), name.c_str(), theType.name().c_str());
    abort();
  }
  if (func.functionParameterSize(true) != 0) {
    fprintf(stderr, "FunctionWithDict: invokeByName<%s>: "
            "function '%s' in type '%s' should have zero "
            "parameters, but has %lu parameters instead!\n",
            typeid(T).name(), name.c_str(), theType.name().c_str(),
            func.functionParameterSize(true));
    abort();
    return;
  }
  ObjectWithDict retobj(typeid(T), &retval);
  func.invoke(&retobj);
}

} // namespace edm

#endif // FWCore_Utilities_FunctionWithDict_h
