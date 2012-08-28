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
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "Reflex/TypeTemplate.h"

namespace edm {

  class FunctionWithDict;
  class MemberWithDict;
  class ObjectWithDict;

  enum TypeNameHandling {
     NoHandling = 0,
     Final = Reflex::FINAL,
     Qualified = Reflex::QUALIFIED,
     FinalQualified = Reflex::FINAL|Reflex::QUALIFIED,
     Scoped = Reflex::SCOPED,
     FinalScoped = Reflex::FINAL|Reflex::SCOPED,
     QualifiedScoped = Reflex::QUALIFIED|Reflex::SCOPED,
     FinalQualifiedScoped = Reflex::FINAL|Reflex::QUALIFIED|Reflex::SCOPED
  };

  enum TypeMemberQuery {
    InheritedDefault = Reflex::INHERITEDMEMBERS_DEFAULT,
    InheritedNo = Reflex::INHERITEDMEMBERS_NO,
    InheritedAlso = Reflex::INHERITEDMEMBERS_ALSO
  };

  enum TypeModifiers {
     NoMod = 0,
     Const = Reflex::CONST,
     Reference = Reflex::REFERENCE,
     ConstReference = Reflex::CONST|Reflex::REFERENCE
  };

  class TypeWithDict {
  public:

    TypeWithDict() : type_() {}

    explicit TypeWithDict(std::type_info const& t);

    template <typename T>
    explicit TypeWithDict(T const& t) : type_(Reflex::Type::ByTypeInfo(typeid(t))) {
    }

    TypeWithDict(TypeWithDict const& type, TypeModifiers modifiers) : type_(Reflex::Type(type.type_, modifiers)) {
    }

    static TypeWithDict
    byName(std::string const& className);

    // Print out the name of the type, using the dictionary class name.
    void print(std::ostream& os) const;

    std::string className() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

    bool hasDictionary() const;

    bool isComplete() const;

    bool hasProperty(std::string const& property) const;

    bool isClass() const {
      return type_.IsClass();
    }

    bool isConst() const {
      return type_.IsConst();
    }

    bool isEnum() const {
      return type_.IsEnum();
    }

    bool isFundamental() const {
      return type_.IsFundamental();
    }

    bool isPointer() const {
      return type_.IsPointer();
    }

    bool isReference() const {
      return type_.IsReference();
    }

    bool isTemplateInstance() const {
      return type_.IsTemplateInstance();
    }

    bool isTypedef() const {
      return type_.IsTypedef();
    }

    TypeWithDict finalType() const {
      return TypeWithDict(type_.FinalType());
    }

    TypeWithDict returnType() const {
      return TypeWithDict(type_.ReturnType());
    }

    TypeWithDict toType() {
      return TypeWithDict(type_.ToType());
    }

    std::string propertyValueAsString(std::string const& property) const;

    TypeWithDict functionParameterAt(size_t index) const {
      return TypeWithDict(type_.FunctionParameterAt(index));
    }

    size_t functionParameterSize() const {
      return type_.FunctionParameterSize();
    }

    TypeWithDict subTypeAt(size_t index) const {
      return TypeWithDict(type_.SubTypeAt(index));
    }

    TypeWithDict templateArgumentAt(size_t index) const {
      return TypeWithDict(type_.TemplateArgumentAt(index));
    }

    ObjectWithDict construct() const;

    ObjectWithDict construct(TypeWithDict const& type, std::vector<void *> const& args) const;

    void destruct(void * address, bool dealloc = true) const {
      type_.Destruct(address, dealloc);
    }

    void const* pointerToContainedType(void const* ptr, TypeWithDict const& containedType) const;

    template <typename T>
    void invokeByName(T& obj, std::string const& name) const {
      Reflex::Member theFunction = type_.FunctionMemberByName(name);
      theFunction.Invoke(obj);
    }

    std::string name(int mod = 0) const {
      return type_.Name(mod);
    }

    void* id() const {
      return type_.Id();
    }

    void* allocate() const {
      return type_.Allocate();
    }

    void deallocate(void* instance) const {
      type_.Deallocate(instance);
    }

#ifndef __GCCXML__
    explicit operator bool() const;
#endif
    
    bool operator<(TypeWithDict const& b) const { return type_ < b.type_; }

    bool operator==(TypeWithDict const& b) const {return type_ == b.type_;}

    bool isEquivalentTo(TypeWithDict const& other) const {
      return type_.IsEquivalentTo(other.type_);
    }

    std::type_info const& typeInfo() const {
      return type_.TypeInfo();
    }

    MemberWithDict dataMemberByName(std::string const& member) const;

    FunctionWithDict functionMemberByName(std::string const& member) const;

    FunctionWithDict functionMemberByName(std::string const& member, TypeWithDict const& signature, int mods, TypeMemberQuery memberQuery) const;

    size_t dataMemberSize() const {
      return type_.DataMemberSize();
    }

    size_t functionMemberSize() const {
      return type_.FunctionMemberSize();
    }

   size_t subTypeSize() const {
      return type_.SubTypeSize();
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

    explicit TypeWithDict(Reflex::Type const& type);

    Reflex::Type type_;
  };

  inline bool operator>(TypeWithDict const& a, TypeWithDict const& b) {
    return b < a;
  }

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

    bool operator==(TypeTemplateWithDict const& other) {
      return typeTemplate_ == other.typeTemplate_;
    }

    std::string name(int mod = 0) const {
      return typeTemplate_.Name(mod);
    }


#ifndef __GCCXML__
    explicit operator bool() const;
#endif
    
  private:
    explicit TypeTemplateWithDict(Reflex::TypeTemplate const& typeTemplate);
    Reflex::TypeTemplate typeTemplate_;
  };

  class TypeBases {
  public:
    explicit TypeBases(TypeWithDict const& type) : type_(type.type_) {}
    Reflex::Base_Iterator begin() const;
    Reflex::Base_Iterator end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
  };

  class TypeDataMembers {
  public:
    explicit TypeDataMembers(TypeWithDict const& type) : type_(type.type_) {}
    Reflex::Member_Iterator begin() const;
    Reflex::Member_Iterator end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
  };

  class TypeFunctionMembers {
  public:
    explicit TypeFunctionMembers(TypeWithDict const& type) : type_(type.type_) {}
    Reflex::Member_Iterator begin() const;
    Reflex::Member_Iterator end() const;
    size_t size() const;
  private:
    Reflex::Type const& type_;
  };
}
#endif
