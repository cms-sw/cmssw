#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "TClass.h"
#include "TClassTable.h"
#include "TDataType.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TInterpreter.h"
#include "TMethodArg.h"
#include "TROOT.h"

#include "boost/thread/tss.hpp"

#include "tbb/concurrent_unordered_map.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <typeinfo>

#include <cxxabi.h>

namespace edm {
   static
   void throwTypeException(std::string const& function, std::string const& typeName) {
      throw Exception(errors::DictionaryNotFound)
         << "Function " << function << ",\n"
         << "no data dictionary found for type:\n\n"
         <<  typeName
         << "\nMost likely the dictionary was never generated,\n"
         << "but it may be that it was generated in the wrong package.\n"
         << "Please add (or move) the specification\n"
         << "<class name=\"whatever\"/>\n"
         << "to the appropriate classes_def.xml file.\n"
         << "If the class is a template instance, you may need\n"
         << "to define a dummy variable of this type in classes.h.\n"
         << "Also, if this class has any transient members,\n"
         << "you need to specify them in classes_def.xml.";

   }

  TypeWithDict
  TypeWithDict::byName(std::string const& name, long property /*= 0L*/) {
    // Note: The property flag should include kIsConstant and
    //       kIsReference if needed since a typeid() expression
    //       ignores those properties, so we must store them
    //       separately.

    typedef tbb::concurrent_unordered_map<std::string, TypeWithDict> TypeMap;
    static TypeMap typeMap;
    TypeMap::const_iterator it = typeMap.find(name);
    if (it != typeMap.end()) {
      return TypeWithDict(it->second, property);
    }
    TClass* theClass = TClass::GetClass(name.c_str());
    if (theClass != nullptr && theClass->GetTypeInfo() != nullptr) {
      return TypeWithDict(theClass, property);
    }
    TType* type = gInterpreter->Type_Factory(name);
    if (!gInterpreter->Type_IsValid(type)) {
      typeMap.insert(std::make_pair(name, TypeWithDict()));
      return TypeWithDict();
    }
    typeMap.insert(std::make_pair(name, TypeWithDict(type, 0L)));
    return TypeWithDict(type, property);
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
    class_(TClass::GetClass(ti)),
    enum_(nullptr),
    dataType_(TDataType::GetDataType(TDataType::GetType(ti))),
    property_(property) {

    if(class_ != nullptr || dataType_ != nullptr) {
      return;
    }

    type_ = gInterpreter->Type_Factory(ti);
    if (!gInterpreter->Type_IsValid(type_)) {
      throwTypeException("TypeWithDict(TType*, property): ", name());
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
      bool valid = gInterpreter->Type_IsValid(ttype);
      if (!valid) {
        throwTypeException("TypeWithDict(TType*, property): ", name());
      }
      ti_ = gInterpreter->Type_TypeInfo(ttype);
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
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr) {
      return true; 
    }
    assert(type_ != nullptr);
    return gInterpreter->Type_Bool(type_);
  }

  bool
  TypeWithDict::hasDictionary() const {
    if (*ti_ == typeid(void)) {
      return true;
    }
    if (ti_->name()[1] == '\0') {
      // returns true for built in types (single character mangled names)
      return true; 
    }
    return (TClassTable::GetDict(*ti_) != nullptr);
  }

  std::type_info const&
  TypeWithDict::typeInfo() const {
    return *ti_;
  }

  std::type_info const&
  TypeWithDict::id() const {
    return *ti_;
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
    return class_ != nullptr;
  }

  bool
  TypeWithDict::isConst() const {
    // Note: We must check the property flags here because typeid() ignores const.
    return (property_ & (long) kIsConstant);
  }

  bool
  TypeWithDict::isArray() const {
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr) {
      return false;
    }
    return (name().back() == ']');
  }

  bool
  TypeWithDict::isEnum() const {
    return enum_ != nullptr;
  }

  bool
  TypeWithDict::isFundamental() const {
    return dataType_ != nullptr;
  }

  bool
  TypeWithDict::isPointer() const {
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr) {
      return false;
    }
    return (name().back() == '*');
  }

  bool
  TypeWithDict::isReference() const {
    // Note: We must check the property flags here too because
    //       typeid() ignores references.
    if (property_ & (long) kIsReference) {
      return true;
    }
    if (type_ == nullptr) {
      return false;
    }
    return gInterpreter->Type_IsReference(type_);
  }

  bool
  TypeWithDict::isTemplateInstance() const {
    if (class_ == nullptr ) {
      return false;
    }
    return (name().back() == '>');
  }

  bool
  TypeWithDict::isTypedef() const {
    if (class_ != nullptr || dataType_ != nullptr) {
      return false;
    }
    assert(type_ != nullptr);
    return gInterpreter->Type_IsTypedef(type_);
  }

  bool
  TypeWithDict::isVirtual() const {
    if (class_ == nullptr) {
      return false;
    }
    return (class_->ClassProperty() & (long) kClassHasVirtual);
  }

  void
  TypeWithDict::print(std::ostream& os) const {
    os << name();
  }

  std::string
  TypeWithDict::cppName() const {
    std::string cName = qualifiedName();
    // Get rid of silly ROOT typedefs
    replaceString(cName, "ULong64_t", "unsigned long long");
    replaceString(cName, "Long64_t", "long long");
    return cName;
  }

  std::string
  TypeWithDict::qualifiedName() const {
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
    return stripNamespace(name());
  }

  std::string
  TypeWithDict::name() const {
    return TypeID(*ti_).className();
  }

  std::string
  TypeWithDict::unscopedNameWithTypedef() const {
    if (type_ == nullptr) {
      return unscopedName();
    }
    return stripNamespace(gInterpreter->Type_QualifiedName(type_));
  }

  std::string
  TypeWithDict::userClassName() const {
    //FIXME: What about const and reference?
    return TypeID(*ti_).userClassName();
  }

  std::string
  TypeWithDict::friendlyClassName() const {
    //FIXME: What about const and reference?
    return TypeID(*ti_).friendlyClassName();
  }

  size_t
  TypeWithDict::size() const {
    if(class_ != nullptr) {
      return class_->GetClassSize();
    }
    if(dataType_ != nullptr) {
      return dataType_->Size();
    }
    assert(type_ != nullptr);
    return gInterpreter->Type_Size(type_);
  }

  size_t
  TypeWithDict::arrayLength() const {
    assert(type_ != nullptr);
    return gInterpreter->Type_ArraySize(type_);
  }

  size_t
  TypeWithDict::arrayDimension() const {
    assert(type_ != nullptr);
    return gInterpreter->Type_ArrayDim(type_);
  }

  size_t
  TypeWithDict::maximumIndex(size_t dim) const {
    assert(type_ != nullptr);
    return gInterpreter->Type_MaxIndex(type_, dim);
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
    if (*ti_ == typeid(void)) {
      return TypeWithDict();
    }
    return TypeWithDict(*ti_);
  }

  TypeWithDict
  TypeWithDict::toType() const {
    if (*ti_ == typeid(void)) {
      return TypeWithDict();
    }
    if(isReference()) {
      long prop = property_ & ~((long) kIsReference);
      return TypeWithDict(*ti_, prop);
    }
    if(isPointer()) {
      std::string newname = name();
      size_t newsize = newname.size() - 1;
      newname.resize(newsize); 
      return byName(newname);
    }
    if(isArray()) {
      std::string newname = name();
      size_t newsize = newname.find_last_of('[');
      newname.resize(newsize); 
      return byName(newname);
    }
    return *this;
  }

  std::string
  TypeWithDict::templateName() const {
    if (!isTemplateInstance()) {
      return "";
    }
    if (name() == "std::string") {
      return std::string("std::basic_string");
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
      return false;
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
      return false;
    }
    if (basety.class_ == nullptr) {
      return false;
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
      throw Exception(errors::LogicError)
        << "Function TypeWithDict::getBaseClassOffset(), type\n"
        << name()
        << "\nis not a class\n";
    }
    if (baseClass.class_ == nullptr) {
      throw Exception(errors::LogicError)
        << "Function TypeWithDict::getBaseClassOffset(), base type\n"
        << name()
        << "\nis not a class\n";
    }
    int offset = class_->GetBaseClassOffset(baseClass.class_);
    return offset;
  }

  int
  TypeWithDict::stringToEnumValue(std::string const& name) const {
    if (enum_ == nullptr) {
      throw Exception(errors::LogicError)
        << "Function TypeWithDict::stringToEnumValue(), type\n"
        << name
        << "\nis not an enum\n";
    }
    TEnumConstant const* ec = enum_->GetConstant(name.c_str());
    if (!ec) {
      throw Exception(errors::LogicError)
        << "Function TypeWithDict::stringToEnumValue(), type\n"
        << name
        << "\nis not an enum constant\n";
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
        throw Exception(errors::LogicError)
          << "Function TypeWithDict::processEnumeration(), enum parent\n"
          << name() 
          << "\nis not a class\n";
      }
      TObject* tobj =
        Tycl.class_->GetListOfEnums()->FindObject(unscopedName().c_str());
      if (tobj == nullptr) {
        throw Exception(errors::LogicError)
          << "Function TypeWithDict::processEnumeration(), enum\n"
          << name()
          << "\nnot found in containing class\n"
          << unscopedName() << "\n";
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
        throw Exception(errors::LogicError)
          << "Function TypeWithDict::processEnumeration(), enum\n"
          << name()
          << "\nnot found in named namespace\n"
          << unscopedName() << "\n";
      }
      enum_ = reinterpret_cast<TEnum*>(tobj);
    } else {
      // Must be a global namespace member.
      //gROOT->GetListOfEnums()->Dump();
      TObject* tobj = gROOT->GetListOfEnums()->FindObject(name().c_str());
      if (tobj == nullptr) {
        throw Exception(errors::LogicError)
          << "Function TypeWithDict::processEnumeration(), enum\n"
          << name()
          << "\nnot found in global namespace\n";
      }
      enum_ = reinterpret_cast<TEnum*>(tobj);
    }
    if (enum_ == nullptr) {
      throw Exception(errors::LogicError)
        << "Function TypeWithDict::processEnumeration(), enum not set for enum type\n"
        << name() << "\n";
    }
  }
  //-------------------------------------------------------------
  //
  //

  // A related free function
  bool
  hasDictionary(std::type_info const& ti) {
    if (ti.name()[1] == '\0') {
      // returns true for built in types (single character mangled names)
      return true; 
    }
    return (TClassTable::GetDict(ti) != nullptr);
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
