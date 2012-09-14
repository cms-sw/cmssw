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

    TypeWithDict(TypeWithDict const& type, TypeModifiers modifiers);

    explicit TypeWithDict(Reflex::Type const& type);

    static TypeWithDict
    byName(std::string const& className);

    // Print out the name of the type, using the dictionary class name.
    void print(std::ostream& os) const;

    std::string className() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

    bool hasDictionary() const;

    bool isComplete() const;

    bool isClass() const;

    bool isConst() const;

    bool isEnum() const;

    bool isFundamental() const;

    bool isPointer() const;

    bool isReference() const;

    bool isStruct() const;

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

    std::string name(int mod = Reflex::FINAL|Reflex::SCOPED) const;

    void* id() const;

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

    Reflex::Type type_;
  };

  inline bool operator<(TypeWithDict const& a, TypeWithDict const& b) {
    return a.typeInfo().before(b.typeInfo());
  }

  inline bool operator==(TypeWithDict const& a, TypeWithDict const& b) {
    return a.typeInfo() == b.typeInfo();
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
