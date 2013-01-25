#ifndef FWCore_Utilities_TypeWithDict_h
#define FWCore_Utilities_TypeWithDict_h

/*----------------------------------------------------------------------
  
TypeWithDict: A unique identifier for a C++ type, with the dictionary information

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <typeinfo>
#include <string>
#include <vector>

#include "TBaseClass.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethod.h"

#include "FWCore/Utilities/interface/IterWithDict.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"

class TBaseClass;
class TDataMember;
class TMethod;
class TMethodArg;

namespace edm {

  class FunctionWithDict;
  class MemberWithDict;
  class ObjectWithDict;

  enum TypeMemberQuery {
    InheritedDefault = Reflex::INHERITEDMEMBERS_DEFAULT,
    InheritedNo = Reflex::INHERITEDMEMBERS_NO,
    InheritedAlso = Reflex::INHERITEDMEMBERS_ALSO
  };

  class TypeWithDict {
  public:

    TypeWithDict();

    explicit TypeWithDict(std::type_info const& t);

    explicit TypeWithDict(std::type_info const& t, Long_t property);

    explicit TypeWithDict(Reflex::Type const& type);

    template <typename T>
    explicit TypeWithDict(T const& t) :
      typeInfo_(&typeid(t)),
      type_(Reflex::Type::ByTypeInfo(typeid(t))),
      class_(TClass::GetClass(typeid(t))),
      dataType_(TDataType::GetDataType(TDataType::GetType(typeid(t)))) {
    }

    explicit TypeWithDict(TypeWithDict const& type, Long_t property);

    explicit TypeWithDict(TClass* type, Long_t property = (Long_t)kIsClass);

    explicit TypeWithDict(TMethodArg* arg);

    static TypeWithDict
    byName(std::string const& className);

    static TypeWithDict
    byName(std::string const& className, Long_t property);

    // Print out the name of the type, using the dictionary class name.
    void print(std::ostream& os) const;

    std::string qualifiedName() const;

    std::string unscopedName() const;

    std::string name() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

    bool hasDictionary() const;

    bool isClass() const;

    bool isConst() const;

    bool isEnum() const;

    bool isFundamental() const;

    bool isPointer() const;

    bool isReference() const;

    bool isTemplateInstance() const;

    bool isTypedef() const;

    TypeWithDict finalType() const;

    TypeWithDict toType() const;

    TypeWithDict nestedType(char const* name) const;

    TypeWithDict nestedType(std::string const& name) const {
      return nestedType(name.c_str());
    }

    TypeWithDict templateArgumentAt(size_t index) const;

    ObjectWithDict construct() const;

    ObjectWithDict construct(TypeWithDict const& type, std::vector<void *> const& args) const;

    void destruct(void * address, bool dealloc = true) const;

    void const* pointerToContainedType(void const* ptr, TypeWithDict const& containedType) const;

    template <typename T>
    void invokeByName(T& obj, std::string const& name) const {
      Reflex::Member theFunction = type_.FunctionMemberByName(name);
      theFunction.Invoke(obj);
    }

    void const* id() const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif
    
    bool isEquivalentTo(TypeWithDict const& other) const;

    std::type_info const& typeInfo() const;

    MemberWithDict dataMemberByName(std::string const& member) const;

    FunctionWithDict functionMemberByName(std::string const& member) const;

    FunctionWithDict functionMemberByName(std::string const& member, TypeWithDict const& signature, int mods, TypeMemberQuery memberQuery) const;

    size_t dataMemberSize() const;

    size_t functionMemberSize() const;

    int stringToEnumValue(std::string const& enumMemberName) const;

    size_t size() const;

    void* allocate() const {
      return new char[size()];
    }

    void deallocate(void * obj) const {
      delete [] static_cast<char *>(obj);
    }

  private:
    friend class BaseWithDict;
    friend class FunctionWithDict;
    friend class MemberWithDict;
    friend class ObjectWithDict;
    friend class TypeBases;
    friend class TypeDataMembers;
    friend class TypeFunctionMembers;
    friend class TypeTemplateWithDict;

    std::type_info const* typeInfo_;
    Reflex::Type type_;
    TClass* class_;
    TDataType* dataType_;
    Long_t property_;
  };

  inline bool operator<(TypeWithDict const& a, TypeWithDict const& b) {
    return a.typeInfo().before(b.typeInfo());
  }

  bool operator==(TypeWithDict const& a, TypeWithDict const& b);

  inline bool operator!=(TypeWithDict const& a, TypeWithDict const& b) {
    return !(a == b);
  }

  std::ostream& operator<<(std::ostream& os, TypeWithDict const& id);

  class TypeTemplateWithDict {
  public:
    TypeTemplateWithDict() : typeTemplate_() {}

    explicit TypeTemplateWithDict(TypeWithDict const& type);

    static TypeTemplateWithDict
    byName(std::string const& templateName, int n);

    std::string name(int mod = 0) const;

    bool operator==(TypeTemplateWithDict const& other) const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif

  private:
    explicit TypeTemplateWithDict(Reflex::TypeTemplate const& typeTemplate);
    Reflex::TypeTemplate typeTemplate_;
  };

  class TypeBases {
  public:
    explicit TypeBases(TypeWithDict const& type) : type_(type.type_), class_(type.class_) {}
    IterWithDict<TBaseClass> begin() const;
    IterWithDict<TBaseClass> end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
    TClass* class_;
  };

  class TypeDataMembers {
  public:
    explicit TypeDataMembers(TypeWithDict const& type) : type_(type.type_), class_(type.class_) {}
    IterWithDict<TDataMember> begin() const;
    IterWithDict<TDataMember> end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
    TClass* class_;
  };

  class TypeFunctionMembers {
  public:
    explicit TypeFunctionMembers(TypeWithDict const& type) : type_(type.type_), class_(type.class_) {}
/*
    IterWithDict<TMethod> begin() const;
    IterWithDict<TMethod> end() const;
*/
    Reflex::Member_Iterator begin() const;
    Reflex::Member_Iterator end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
    TClass* class_;
  };
}
#endif
