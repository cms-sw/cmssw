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
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"

namespace edm {

  class MemberWithDict;
  class ObjectWithDict;

  class TypeWithDict : private TypeIDBase {
  public:

    TypeWithDict() : TypeIDBase(), type_() {}

    explicit TypeWithDict(std::type_info const& t);

    template <typename T>
    explicit TypeWithDict(T const& t) : TypeIDBase(typeid(t)), type_(Reflex::Type::ByTypeInfo(typeid(t))) {
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

    bool isFundamental() const {
      return type_.IsFundamental();
    }

    bool isPointer() const {
      return type_.IsPointer();
    }

    bool isReference() const {
      return type_.IsReference();
    }

    bool isTypedef() const {
      return type_.IsTypedef();
    }

    TypeWithDict toType() {
      return TypeWithDict(type_.ToType());
    }

    std::string propertyValueAsString(std::string const& property) const;

    ObjectWithDict  construct() const;

    void const* pointerToContainedType(void const* ptr, TypeWithDict const& containedType) const;

    template <typename T>
    void invokeByName(T& obj, std::string const& name) const {
      Reflex::Member theFunction = type_.FunctionMemberByName(name);
      theFunction.Invoke(obj);
    }


#ifndef __GCCXML__
    explicit operator bool() const;
#endif
    
    using TypeIDBase::name;

    bool operator<(TypeWithDict const& b) const { return this->TypeIDBase::operator<(b); }

    bool operator==(TypeWithDict const& b) const {return this->TypeIDBase::operator==(b);}

    bool isEquivalentTo(TypeWithDict const& other) const {
      return type_.IsEquivalentTo(other.type_);
    }

    using TypeIDBase::typeInfo;

    MemberWithDict memberByName(std::string const& member) const;

    MemberWithDict dataMemberByName(std::string const& member) const;

    MemberWithDict functionMemberByName(std::string const& member) const;

  private:
    friend class BaseWithDict;
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
  private:
    Reflex::Type const& type_;
  };

  class TypeDataMembers {
  public:
    explicit TypeDataMembers(TypeWithDict const& type) : type_(type.type_) {}
    Reflex::Member_Iterator begin() const;
    Reflex::Member_Iterator end() const;
  private:
    Reflex::Type const& type_;
  };

  class TypeFunctionMembers {
  public:
    explicit TypeFunctionMembers(TypeWithDict const& type) : type_(type.type_) {}
    Reflex::Member_Iterator begin() const;
    Reflex::Member_Iterator end() const;
  private:
    Reflex::Type const& type_;
  };
}
#endif
