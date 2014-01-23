#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "TClass.h"
#include "TDataType.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TInterpreter.h"
#include "TMethodArg.h"
#include "TROOT.h"

#include "boost/thread/tss.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <typeinfo>

#include <cxxabi.h>
#include <iostream>

namespace edm {

  TypeWithDict
  TypeWithDict::byName(std::string const& name, long property /*= 0L*/) {
    // Note: The property flag should include kIsConstant and
    //       kIsReference if needed since a typeid() expression
    //       ignores those properties, so we must store them
    //       separately.
    TType* type = gInterpreter->Type_Factory(name);
    if (!gInterpreter->Type_IsValid(type)) {
      // This can happen if no dictionary has been loaded for the class or enum.
      // If this is a class, trigger the loading of the dictionary.
      TClass* theClass = TClass::GetClass(name.c_str());
      if(theClass == nullptr || theClass->GetTypeInfo() == nullptr) {
        return TypeWithDict();
      }
      return TypeWithDict(theClass, property);
    }
    return TypeWithDict(type, property);
  }

  void
  TypeWithDict::setProperty() {
    // Note: typeid() ignores const and volatile qualifiers, and
    //       cannot see references.
    //       So:
    //            typeid(int) == typeid(int const)
    //                        == typeid(int&)
    //                        == typeid(int const&)
    //
    if (type_ == nullptr) {
      return;
    }
    if (gInterpreter->Type_IsConst(type_)) {
      // Note: This is not possible if created by typeid.
      property_ |= (long) kIsConstant;
    }
    if (gInterpreter->Type_IsReference(type_)) {
      // Note: This is not possible if created by typeid.
      property_ |= (long) kIsReference;
    }
  }

  TypeWithDict::TypeWithDict() :
    ti_(&typeid(void)),
    type_(nullptr),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    property_(0L) {
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& rhs) :
    ti_(rhs.ti_),
    type_(rhs.type_),
    class_(rhs.class_),
    enum_(rhs.enum_),
    dataType_(rhs.dataType_),
    property_(rhs.property_) {
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& type, long prop) :
    ti_(type.ti_),
    type_(type.type_),
    class_(type.class_),
    enum_(type.enum_),
    dataType_(type.dataType_),
    property_(type.property_) {
    // Unconditionally modifies const and reference
    // properties, and only those properties.
    property_ &= ~((long) kIsConstant | (long) kIsReference);
    if (prop & kIsConstant) {
      property_ |= kIsConstant;
    }
    if (prop & kIsReference) {
      property_ |= kIsReference;
    }
  }

  TypeWithDict&
  TypeWithDict::operator=(TypeWithDict const& rhs) {
    if (this != &rhs) {
      ti_ = rhs.ti_;
      type_ = rhs.type_;
      class_ = rhs.class_;
      enum_ = rhs.enum_;
      dataType_ = rhs.dataType_;
      property_ = rhs.property_;
    }
    return *this;
  }

  TypeWithDict::TypeWithDict(std::type_info const& ti, long property /*= 0L*/) :
    ti_(&ti),
    type_(nullptr),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    property_(property) {

    type_ = gInterpreter->Type_Factory(ti);
    if (!gInterpreter->Type_TypeInfo(type_)) {
      // This can happen if no dictionary has been loaded for the class or enum.
      // If this is a class, trigger the loading of the dictionary.
      class_ = TClass::GetClass(ti);
    }
    if (!gInterpreter->Type_TypeInfo(type_)) {
      // FIXME: Replace this with an exception!
      fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
              "Type_TypeInfo returns nullptr!\n");
      abort();
    }
    dataType_ = TDataType::GetDataType(TDataType::GetType(ti));
    if (class_ == nullptr &&
        !gInterpreter->Type_IsFundamental(type_) &&
        !gInterpreter->Type_IsEnum(type_)) {
      // Must be a class, struct, or union.
      class_ = TClass::GetClass(ti);
    }
    if (gInterpreter->Type_IsEnum(type_)) {
      processEnumeration();
    }
  }

  TypeWithDict::TypeWithDict(TClass* cl, long property /*= 0L*/) :
    ti_(&typeid(void)),
    type_(nullptr),
    class_(cl),
    enum_(nullptr),
    dataType_(nullptr),
    property_((long) kIsClass | property) {

    ti_ = cl->GetTypeInfo();
    type_ = gInterpreter->Type_Factory(*cl->GetTypeInfo());
  }

  TypeWithDict::TypeWithDict(TMethodArg* arg, long property /*= 0L*/) :
    TypeWithDict(byName(arg->GetTypeName(), arg->Property() | property)) {
  }

  TypeWithDict::TypeWithDict(TType* ttype, long property /*= 0L*/) :
    ti_(&typeid(void)),
    type_(ttype),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    property_(property) {

    if (!ttype) {
      return;
    }
    {
      std::type_info const* info = gInterpreter->Type_TypeInfo(ttype);
      if (!info) {
        // FIXME: Replace this with an exception!
        fprintf(stderr, "TypeWithDict(TType*, property): "
                "Type_TypeInfo returns nullptr!\n");
        abort();
      }
      ti_ = info;
    }
    if (gInterpreter->Type_IsConst(ttype)) {
      // Note: This is not possible if created by typeid.
      property_ |= (long) kIsConstant;
    }
    if (gInterpreter->Type_IsReference(ttype)) {
      // Note: This is not possible if created by typeid.
      property_ |= (long) kIsReference;
    }
    dataType_ = TDataType::GetDataType(TDataType::GetType(*ti_));
    if (!gInterpreter->Type_IsFundamental(ttype) &&
        !gInterpreter->Type_IsEnum(ttype)) {
      // Must be a class, struct, or union.
      class_ = TClass::GetClass(*ti_);
    }
    if (gInterpreter->Type_IsEnum(type_)) {
      processEnumeration();
    }
  }

  TypeWithDict::operator bool() const {
    if (*ti_ == typeid(void)) {
      return false;
    }
    if (type_ == nullptr) {
      // FIXME: Replace this with an exception!
      fprintf(stderr, "TypeWithDict::operator bool(): "
              "type_ is nullptr!\n");
      abort();
    }
    return gInterpreter->Type_Bool(type_);
  }

  bool
  TypeWithDict::hasDictionary() const {
    if (*ti_ == typeid(void)) {
      return true;
    }
    if (type_ == nullptr) {
      // FIXME: Replace this with an exception!
      fprintf(stderr, "TypeWithDict::hasDictionary(): "
              "type_ is nullptr!\n");
      abort();
    }
    return gInterpreter->Type_Bool(type_);
  }

  std::type_info const&
  TypeWithDict::typeInfo() const {
    return *ti_;
  }

  std::type_info const&
  TypeWithDict::id() const {
    return *ti_;
  }

  TType*
  TypeWithDict::getType() const {
    return type_;
  }

  TClass*
  TypeWithDict::getClass() const {
    return class_;
  }

  TEnum*
  TypeWithDict::getEnum() const {
    return enum_;
  }

  TDataType*
  TypeWithDict::getDataType() const {
    return dataType_;
  }

  long
  TypeWithDict::getProperty() const {
    return property_;
  }

  bool
  TypeWithDict::isClass() const {
    // Note: This really means is class, struct, or union.
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsClass(type_);
  }

  bool
  TypeWithDict::isConst() const {
    // Note: We must check the property flags here too because
    //       typeid() ignores const.
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsConst(type_) || (property_ & (long) kIsConstant);
  }

  bool
  TypeWithDict::isArray() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsArray(type_);
  }

  bool
  TypeWithDict::isEnum() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsEnum(type_);
  }

  bool
  TypeWithDict::isFundamental() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsFundamental(type_);
  }

  bool
  TypeWithDict::isPointer() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsPointer(type_);
  }

  bool
  TypeWithDict::isReference() const {
    // Note: We must check the property flags here too because
    //       typeid() ignores references.
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsReference(type_) ||
           (property_ & (long) kIsReference);
  }

  bool
  TypeWithDict::isTemplateInstance() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsTemplateInstance(type_);
  }

  bool
  TypeWithDict::isTypedef() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsTypedef(type_);
  }

  bool
  TypeWithDict::isVirtual() const {
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsVirtual(type_);
  }

  void
  TypeWithDict::print(std::ostream& os) const {
    os << name();
  }

  std::string
  TypeWithDict::qualifiedName() const {
    if (type_ == nullptr) {
      return "undefined";
    }
    std::string qname(name());
    if (isConst()) {
      qname = "const " + qname;
    }
    if (isReference()) {
      qname += '&';
    }
    return qname;
  }

  std::string
  TypeWithDict::unscopedName() const {
    if (type_ == nullptr) {
      return "undefined";
    }
    return stripNamespace(name());
  }

  std::string
  TypeWithDict::name() const {
    if (type_ == nullptr) {
      return "undefined";
    }
    return TypeID(*ti_).className();
  }

  std::string
  TypeWithDict::unscopedNameWithTypedef() const {
    if (type_ == nullptr) {
      return "undefined";
    }
    return stripNamespace(gInterpreter->Type_QualifiedName(type_));
  }

  std::string
  TypeWithDict::userClassName() const {
    //FIXME: What about const and reference?
    if (type_ == nullptr) {
      return "undefined";
    }
    return TypeID(*ti_).userClassName();
  }

  std::string
  TypeWithDict::friendlyClassName() const {
    //FIXME: What about const and reference?
    if (type_ == nullptr) {
      return "undefined";
    }
    return TypeID(*ti_).friendlyClassName();
  }

  size_t
  TypeWithDict::size() const {
    if (type_ == nullptr) {
      return 0;
    }
    return gInterpreter->Type_Size(type_);
  }

  size_t
  TypeWithDict::arrayLength() const {
    if (type_ == nullptr) {
      return 0;
    }
    return gInterpreter->Type_ArraySize(type_);
  }

  size_t
  TypeWithDict::dataMemberSize() const {
    if (class_ != nullptr) {
      return class_->GetListOfDataMembers()->GetSize();
    }
    if (enum_ != nullptr) {
      return enum_->GetConstants()->GetSize();
    }
    return 0;
  }

  size_t
  TypeWithDict::functionMemberSize() const {
    if (class_ != nullptr) {
      return class_->GetListOfMethods()->GetSize();
    }
    return 0;
  }

  void const*
  TypeWithDict::pointerToBaseType(void const* ptr, TypeWithDict const& derivedType) const {
    if (this->ti_ == derivedType.ti_ || *this->ti_ == *derivedType.ti_) {
      return ptr;
    }
    int offset = derivedType.getBaseClassOffset(*this);
    if (offset < 0) {
      return nullptr;
    }
    return static_cast<char const*>(ptr) + offset;
  }

  void const*
  TypeWithDict::pointerToContainedType(void const* ptr, TypeWithDict const& derivedType) const {
    return pointerToBaseType(ptr, derivedType);
  }

  TypeWithDict
  TypeWithDict::nestedType(char const* nestedName) const {
    return byName(name() + "::" + nestedName);
  }

  TypeWithDict
  TypeWithDict::nestedType(std::string const& nestedName) const {
    return byName(name() + "::" + nestedName);
  }

  MemberWithDict
  TypeWithDict::dataMemberByName(std::string const& member) const {
    if (class_ != nullptr) {
      return MemberWithDict(class_->GetDataMember(member.c_str()));
    }
    if (enum_ != nullptr) {
      TClass* cl = enum_->GetClass();
      return MemberWithDict(cl->GetDataMember(member.c_str()));
    }
    return MemberWithDict();
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& member) const {
    if (class_ == nullptr) {
      return FunctionWithDict();
    }
    TMethod* meth = reinterpret_cast<TMethod*>(
      class_->GetListOfMethods()->FindObject(member.c_str()));
    if (meth == nullptr) {
      return FunctionWithDict();
    }
    return FunctionWithDict(meth);
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& name, std::string const& proto, bool isConst) const {
    if (class_ == nullptr) {
      return FunctionWithDict();
    }
    TMethod* meth = class_->GetMethodWithPrototype(name.c_str(), proto.c_str(), /*objectIsConst=*/isConst, /*mode=*/ROOT::kExactMatch);
    if (meth == nullptr) {
      return FunctionWithDict();
    }
    return FunctionWithDict(meth);
  }

  TypeWithDict
  TypeWithDict::finalType() const {
    TypeWithDict ret;
    if (type_ == nullptr) {
      return ret;
    }
    TType* ty = gInterpreter->Type_FinalType(type_);
    if (ty == nullptr) {
      return ret;
    }
    std::type_info const* ti = gInterpreter->Type_TypeInfo(ty);
    if (ti == nullptr) {
      return ret;
    }
    ret = TypeWithDict(*ti);
    return ret;
  }

  TypeWithDict
  TypeWithDict::toType() const {
    TypeWithDict ret;
    if (type_ == nullptr) {
      return ret;
    }
    TType* ty = gInterpreter->Type_ToType(type_);
    if (ty == nullptr) {
      return ret;
    }
    std::type_info const* ti = gInterpreter->Type_TypeInfo(ty);
    if (ti == nullptr) {
      return ret;
    }
    ret = TypeWithDict(*ti);
    return ret;
  }

  std::string
  TypeWithDict::templateName() const {
    if (!isTemplateInstance()) {
      return "";
    }
    std::string templateName(name());
    auto begin = templateName.find('<');
    assert(begin != std::string::npos);
    auto end = templateName.rfind('<');
    assert(end != std::string::npos);
    assert(begin <= end);
    if (begin < end) {
      int depth = 1;
      for (auto idx = begin + 1; idx <= end; ++idx) {
        char c = templateName[idx];
        if (c == '<') {
          if (depth == 0) {
            begin = idx;
          }
          ++depth;
        }
        else if (c == '>') {
          --depth;
          assert(depth >= 0);
        }
      }
    }
    return templateName.substr(0, begin);
  }

  TypeWithDict
  TypeWithDict::templateArgumentAt(size_t index) const {
    std::string className(unscopedName());
    auto begin = className.find('<');
    if (begin == std::string::npos) {
      return TypeWithDict();
    }
    ++begin;
    auto end = className.rfind('>');
    assert(end != std::string::npos);
    assert(begin < end);
    int depth = 0;
    size_t argCount = 0;
    for (auto idx = begin; idx < end; ++idx) {
      char c = className[idx];
      if (c == '<') {
        ++depth;
      }
      else if (c == '>') {
        --depth;
        assert(depth >= 0);
      }
      else if ((depth == 0) && (c == ',')) {
        if (argCount < index) {
          begin = idx + 1;
          ++argCount;
        }
        else {
          end = idx;
          break;
        }
      }
    }
    assert(depth == 0);
    if (argCount < index) {
      return TypeWithDict();
    }
    return byName(className.substr(begin, end - begin));
  }

  bool
  TypeWithDict::hasBase(std::string const& basename) const {
    if (class_ == nullptr) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::hasBase(basename): "
              "type is not a class!\n");
      abort();
    }
    TClass* cl = class_->GetBaseClass(basename.c_str());
    if (cl != nullptr) {
      return true;
    }
    return false;
  }

  bool
  TypeWithDict::hasBase(TypeWithDict const& basety) const {
    if (class_ == nullptr) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::hasBase(TypeWithDict const&): "
              "type is not a class!\n");
      abort();
    }
    if (basety.class_ == nullptr) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::hasBase(TypeWithDict const&): "
              "basety is not a class!\n");
      abort();
    }
    TClass* cl = class_->GetBaseClass(basety.name().c_str());
    if (cl != nullptr) {
      return true;
    }
    return false;
  }

  int
  TypeWithDict::getBaseClassOffset(TypeWithDict const& baseClass) const {
    if (class_ == nullptr) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::getBaseClassOffset(TypeWithDict const&): "
              "type is not a class!\n");
      abort();
    }
    if (baseClass.class_ == nullptr) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::getBaseClassOffset(TypeWithDict const&): "
              "baseClass is not a class!\n");
      abort();
    }
    int offset = class_->GetBaseClassOffset(baseClass.class_);
    return offset;
  }

  int
  TypeWithDict::stringToEnumValue(std::string const& name) const {
    if (enum_ == nullptr) {
      fprintf(stderr, "TypeWithDict::stringToEnumValue(name): "
              "type is not an enum!\n");
      abort();
    }
    TEnumConstant const* ec = enum_->GetConstant(name.c_str());
    if (!ec) {
      // FIXME: Turn this into a throw!
      fprintf(stderr, "TypeWithDict::stringToEnumValue(name): "
              "no enum constant named '%s'!\n", name.c_str());
      abort();
    }
    return static_cast<int>(ec->GetValue());
  }

  void*
  TypeWithDict::allocate() const {
    return new char[size()];
  }

  void
  TypeWithDict::deallocate(void* address) const {
    delete[] reinterpret_cast<char*>(address);
  }

  ObjectWithDict
  TypeWithDict::construct() const {
    if (class_ != nullptr) {
      return ObjectWithDict(*this, class_->New());
    }
    return ObjectWithDict(*this, new char[size()]);
  }

  void
  TypeWithDict::destruct(void* address, bool dealloc) const {
    if (class_ != nullptr) {
      class_->Destructor(address, !dealloc);
      return;
    }
    if (dealloc) {
      delete[] reinterpret_cast<char*>(address);
    }
  }

  void
  TypeWithDict::processEnumeration() {
    TType* TTy = gInterpreter->Type_GetParent(type_);
    if (TTy != nullptr) {
      // The enum is a class member.
      TypeWithDict Tycl(*gInterpreter->Type_TypeInfo(TTy));
      if (Tycl.class_ == nullptr) {
        // FIXME: Replace this with an exception!
        fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
                "Enum parent is not a class!\n");
        abort();
      }
      TObject* tobj =
        Tycl.class_->GetListOfEnums()->FindObject(unscopedName().c_str());
      if (tobj == nullptr) {
        // FIXME: Replace this with an exception!
        fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
                "Enum not found in containing class!\n");
        abort();
      }
      enum_ = reinterpret_cast<TEnum*>(tobj);
    } else if (name().size() != unscopedName().size()) {
      // Must be a namespace member.
      assert(name().size() >= unscopedName().size() + 2);
      std::string theNamespace(name().substr(0, name().size() - unscopedName().size() - 2));
      TClass* cl = TClass::GetClass(theNamespace.c_str());
      assert(cl != nullptr);
      TObject* tobj = cl->GetListOfEnums()->FindObject(unscopedName().c_str());
      if (tobj == nullptr) {
        // FIXME: Replace this with an exception!
        fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
                "Enum not found in named namespace!\n");
        abort();
      }
      enum_ = reinterpret_cast<TEnum*>(tobj);
    } else {
      // Must be a global namespace member.
      //gROOT->GetListOfEnums()->Dump();
      TObject* tobj = gROOT->GetListOfEnums()->FindObject(name().c_str());
      if (tobj == nullptr) {
        // FIXME: Replace this with an exception!
        fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
                "Enum not found in global namespace!\n");
        abort();
      }
      enum_ = reinterpret_cast<TEnum*>(tobj);
    }
    if (enum_ == nullptr) {
      // FIXME: Replace this with an exception!
      fprintf(stderr, "TypeWithDict(std::type_info const&, long): "
              "enum_ not set for enum type!\n");
      abort();
    }
  }
  //-------------------------------------------------------------
  //
  //

  // A related free function
  bool
  hasDictionary(std::type_info const& ti) {
    return gInterpreter->Type_TypeInfo(gInterpreter->Type_Factory(ti));
  }

  bool
  operator==(TypeWithDict const& a, TypeWithDict const& b) {
    return a.typeInfo() == b.typeInfo();
  }

  std::ostream&
  operator<<(std::ostream& os, TypeWithDict const& ty) {
    ty.print(os);
    return os;
  }

  //-------------------------------------------------------------
  //
  //

  TypeBases::TypeBases(TypeWithDict const& type) :
    type_(type.type_),
    class_(type.class_) {
  }

  IterWithDict<TBaseClass>
  TypeBases::begin() const {
    if (class_ == nullptr) {
      return IterWithDict<TBaseClass>();
    }
    return IterWithDict<TBaseClass>(class_->GetListOfBases());
  }

  IterWithDict<TBaseClass>
  TypeBases::end() const {
    return IterWithDict<TBaseClass>();
  }

  size_t
  TypeBases::size() const {
    if (class_ == nullptr) {
      return 0;
    }
    return class_->GetListOfBases()->GetSize();
  }

  //-------------------------------------------------------------
  //
  //

  TypeDataMembers::TypeDataMembers(TypeWithDict const& type) :
    type_(type.type_),
    class_(type.class_) {
  }

  IterWithDict<TDataMember>
  TypeDataMembers::begin() const {
    if (class_ == nullptr) {
      return IterWithDict<TDataMember>();
    }
    return IterWithDict<TDataMember>(class_->GetListOfDataMembers());
  }

  IterWithDict<TDataMember>
  TypeDataMembers::end() const {
    return IterWithDict<TDataMember>();
  }

  size_t
  TypeDataMembers::size() const {
    if (class_ == nullptr) {
      return 0;
    }
    return class_->GetListOfDataMembers()->GetSize();
  }

  //-------------------------------------------------------------
  //
  //

  TypeFunctionMembers::TypeFunctionMembers(TypeWithDict const& type) :
    type_(type.type_),
    class_(type.class_) {
  }

  IterWithDict<TMethod>
  TypeFunctionMembers::begin() const {
    if (class_ == nullptr) {
      return IterWithDict<TMethod>();
    }
    return IterWithDict<TMethod>(class_->GetListOfMethods());
  }

  IterWithDict<TMethod>
  TypeFunctionMembers::end() const {
    return IterWithDict<TMethod>();
  }

  size_t
  TypeFunctionMembers::size() const {
    if (class_ == nullptr) {
      return 0;
    }
    return class_->GetListOfMethods()->GetSize();
  }

} // namespace edm
