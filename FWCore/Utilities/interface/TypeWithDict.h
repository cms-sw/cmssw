#ifndef FWCore_Utilities_TypeWithDict_h
#define FWCore_Utilities_TypeWithDict_h

/*----------------------------------------------------------------------

TypeWithDict: A unique identifier for a C++ type, with the dictionary information

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

----------------------------------------------------------------------*/
#include "FWCore/Utilities/interface/IterWithDict.h"

#include "TBaseClass.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TEnum.h"
#include "TMethod.h"
#include "TMethodArg.h"

#include <iosfwd>
#include <string>
#include <typeinfo>
#include <vector>

class TType;

namespace edm {

class FunctionWithDict;
class MemberWithDict;
class ObjectWithDict;

class TypeBases;
class TypeDataMembers;
class TypeFunctionMembers;

class TypeWithDict {
  friend class TypeBases;
  friend class TypeDataMembers;
  friend class TypeFunctionMembers;
private:
  std::type_info const* ti_;
  TType* type_;
  TClass* class_;
  TEnum* enum_;
  TDataType* dataType_;
  long property_;
private:
  void setProperty();
  void processEnumeration();
public:
  static TypeWithDict byName(std::string const& name, long property = 0L);
public:
  TypeWithDict();
  TypeWithDict& operator=(TypeWithDict const&);
  TypeWithDict(TypeWithDict const&);
  // This copy constructor is for clearing const and reference.
  explicit TypeWithDict(TypeWithDict const&, long property);
  explicit TypeWithDict(std::type_info const&, long property = 0L);
  explicit TypeWithDict(TClass* type, long property = 0L);
  explicit TypeWithDict(TMethodArg* arg, long property = 0L);
  explicit TypeWithDict(TType* type, long property = 0L);
  //template<typename T> TypeWithDict() : TypeWithDict(typeid(T)) {}
  explicit operator bool() const;
  bool hasDictionary() const;
  std::type_info const& typeInfo() const;
  std::type_info const& id() const;
  TType* getType() const;
  TClass* getClass() const;
  TEnum* getEnum() const;
  TDataType* getDataType() const;
  long getProperty() const;
  bool isClass() const;
  bool isConst() const;
  bool isArray() const;
  bool isEnum() const;
  bool isFundamental() const;
  bool isPointer() const;
  bool isReference() const;
  bool isTemplateInstance() const;
  bool isTypedef() const;
  bool isVirtual() const;
  std::string qualifiedName() const;
  std::string unscopedName() const;
  std::string name() const;
  std::string unscopedNameWithTypedef() const;
  std::string userClassName() const;
  std::string friendlyClassName() const;
  std::string templateName() const;
  size_t size() const;
  size_t arrayLength() const;
  size_t dataMemberSize() const;
  size_t functionMemberSize() const;
  MemberWithDict dataMemberByName(std::string const&) const;
  // Note: Used only by FWCore/Modules/src/EventContentAnalyzer.cc
  FunctionWithDict functionMemberByName(std::string const&) const;
  // Note: Used only by Fireworks/Core/src/FWModelContextMenuHandler.cc:262
  //FunctionWithDict functionMemberByName(std::string const& name, TypeWithDict const& signature, int mods, TypeMemberQuery memberQuery) const;
  // Note: Used only by CondFormats/PhysicsToolsObjects/src/MVAComputer.cc
  FunctionWithDict functionMemberByName(std::string const& name, std::string const& proto, bool isConst) const;
  TypeWithDict nestedType(char const*) const;
  TypeWithDict nestedType(std::string const&) const;
  TypeWithDict finalType() const;
  TypeWithDict toType() const;
  void print(std::ostream& os) const;
  bool hasBase(std::string const&) const;
  bool hasBase(TypeWithDict const& basety) const;
  int getBaseClassOffset(TypeWithDict const& baseClass) const;
  TypeWithDict templateArgumentAt(size_t index) const;
  void const* pointerToBaseType(void const* ptr, TypeWithDict const& derivedType) const;
  void const* pointerToContainedType(void const* ptr, TypeWithDict const& derivedType) const;
  int stringToEnumValue(std::string const&) const;
  void* allocate() const;
  void deallocate(void* address) const;
  ObjectWithDict construct() const;
  void destruct(void* address, bool dealloc = true) const;
};

// A related free function
bool hasDictionary(std::type_info const&);

inline bool operator<(TypeWithDict const& a, TypeWithDict const& b) {
  return a.typeInfo().before(b.typeInfo());
}

bool operator==(TypeWithDict const& a, TypeWithDict const& b);

inline bool operator!=(TypeWithDict const& a, TypeWithDict const& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os, TypeWithDict const& id);

class TypeBases {
private:
  TType* type_;
  TClass* class_;
public:
  explicit TypeBases(TypeWithDict const&);
  IterWithDict<TBaseClass> begin() const;
  IterWithDict<TBaseClass> end() const;
  size_t size() const;
};

class TypeDataMembers {
private:
  TType* type_;
  TClass* class_;
public:
  explicit TypeDataMembers(TypeWithDict const&);
  IterWithDict<TDataMember> begin() const;
  IterWithDict<TDataMember> end() const;
  size_t size() const;
};

class TypeFunctionMembers {
private:
  TType* type_;
  TClass* class_;
public:
  explicit TypeFunctionMembers(TypeWithDict const&);
  IterWithDict<TMethod> begin() const;
  IterWithDict<TMethod> end() const;
  size_t size() const;
};

} // namespace edm

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#endif // FWCore_Utilities_TypeWithDict_h
