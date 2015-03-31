#ifndef FWCore_Utilities_TypeWithDict_h
#define FWCore_Utilities_TypeWithDict_h

/*----------------------------------------------------------------------

TypeWithDict: A unique identifier for a C++ type, with the dictionary information

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

----------------------------------------------------------------------*/
#include "FWCore/Utilities/interface/IterWithDict.h"
#include "FWCore/Utilities/interface/value_ptr.h"

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
  friend bool operator==(TypeWithDict const&, std::type_info const&);
  typedef enum{} dummyType; // Tag for valid type, but no type_info information
  typedef dummyType** invalidType; // Tag for invalid type
private:
  std::type_info const* ti_;
  TClass* class_;
  TEnum* enum_;
  TDataType* dataType_;
  value_ptr<std::vector<size_t> > arrayDimensions_;
  long property_;
public:
  static TypeWithDict byName(std::string const& name);
private:
  static TypeWithDict byName(std::string const& name, long property);
public:
  TypeWithDict();
  TypeWithDict(TypeWithDict const&);
  explicit TypeWithDict(TClass* type);
  explicit TypeWithDict(TEnum* type);
  explicit TypeWithDict(std::type_info const&);
  explicit TypeWithDict(TMethodArg* arg);
private:
  explicit TypeWithDict(std::type_info const&, long property);
  explicit TypeWithDict(TClass* type, long property);
  explicit TypeWithDict(TEnum* type, long property);
  explicit TypeWithDict(TMethodArg* arg, long property);
public:
  TypeWithDict& operator=(TypeWithDict const&);
  TypeWithDict& stripConstRef();
  explicit operator bool() const;
  std::type_info const& typeInfo() const;
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
  std::string cppName() const;
  std::string unscopedName() const;
  std::string name() const;
  std::string userClassName() const;
  std::string friendlyClassName() const;
  std::string templateName() const;
  size_t size() const;
  size_t arrayLength() const;
  size_t arrayDimension() const;
  size_t maximumIndex(size_t dim) const;
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
  return a.name() < b.name();
}

bool operator==(TypeWithDict const& a, TypeWithDict const& b);

inline bool operator!=(TypeWithDict const& a, TypeWithDict const& b) {
  return !(a == b);
}

bool operator==(TypeWithDict const& a, std::type_info const& b);

inline bool operator!=(TypeWithDict const& a, std::type_info const& b) {
  return !(a == b);
}

inline bool operator==(std::type_info const& a, TypeWithDict const& b) {
  return b == a;
}

inline bool operator!=(std::type_info const& a, TypeWithDict const& b) {
  return !(b == a);
}

std::ostream& operator<<(std::ostream& os, TypeWithDict const& id);

class TypeBases {
private:
  TClass* class_;
public:
  explicit TypeBases(TypeWithDict const&);
  IterWithDict<TBaseClass> begin() const;
  IterWithDict<TBaseClass> end() const;
  size_t size() const;
};

class TypeDataMembers {
private:
  TClass* class_;
public:
  explicit TypeDataMembers(TypeWithDict const&);
  IterWithDict<TDataMember> begin() const;
  IterWithDict<TDataMember> end() const;
  size_t size() const;
};

class TypeFunctionMembers {
private:
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
