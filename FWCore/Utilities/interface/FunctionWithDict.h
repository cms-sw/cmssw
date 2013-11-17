#ifndef FWCore_Utilities_FunctionWithDict_h
#define FWCore_Utilities_FunctionWithDict_h

/*----------------------------------------------------------------------

FunctionWithDict:  A holder for a class member function

----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/IterWithDict.h"

#include "TMethod.h"
#include "TMethodArg.h"

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
  void invoke(const ObjectWithDict& obj, ObjectWithDict* ret = nullptr, const std::vector<void*>& values = std::vector<void*>()) const;
  void invoke(ObjectWithDict* ret = nullptr, const std::vector<void*>& values = std::vector<void*>()) const;
  IterWithDict<TMethodArg> begin() const;
  IterWithDict<TMethodArg> end() const;
};

} // namespace edm

#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

/// Call a static function of class T, derived from the type
/// of the return value, by name with no arguments.
template<typename T>
inline
void
invokeByName(T& retval, const std::string& name)
{
  TypeWithDict theType(typeid(T));
  FunctionWithDict func = theType.functionMemberByName(name);
  if (!bool(func)) {
    return;
  }
  if (func.functionParameterSize(true) != 0) {
    // FIXME: We should throw or write an error message here!
    return;
  }
  ObjectWithDict retobj(typeid(T), &retval);
  func.invoke(retobj);
}

} // namespace edm

#endif // FWCore_Utilities_FunctionWithDict_h
