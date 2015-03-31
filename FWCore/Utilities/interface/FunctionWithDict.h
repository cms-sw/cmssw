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

#endif // FWCore_Utilities_FunctionWithDict_h
