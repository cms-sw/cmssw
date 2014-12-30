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
#include "TRealData.h"
#include "TROOT.h"

#include "boost/thread/tss.hpp"

#include "tbb/concurrent_unordered_map.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <typeinfo>

#include <cxxabi.h>

//#include <iostream>
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
  TypeWithDict::byName(std::string const& name) {
    // This is a public static function.
    return TypeWithDict::byName(name, 0L);
  }

  TypeWithDict
  TypeWithDict::byName(std::string const& name, long property) {
    // This is a private static function.

    typedef tbb::concurrent_unordered_map<std::string, TypeWithDict> TypeMap;
    static TypeMap typeMap;
    static std::string const constPrefix("const ");
    static std::string const constSuffix(" const");
    static size_t const constPrefixSize(constPrefix.size());
    static size_t const constSuffixSize(constSuffix.size());

    // Handle references
    if(name.back() == '&') {
      assert(property == 0L);
      property |= kIsReference;
      return byName(name.substr(0, name.size() - 1), property);
    }

    // Handle const qualifier
    if(name.size() > constSuffixSize && name.back() != '*') {
      if(name.substr(0, constPrefixSize) == constPrefix) {
        property |= kIsConstant;
        return byName(name.substr(constPrefixSize), property);
      }
      if(name.substr(name.size() - constSuffixSize) == constSuffix) {
        property |= kIsConstant;
        return byName(name.substr(0, name.size() - constSuffixSize), property);
      }
    }

    // Handle pointers
    if(name.back() == '*') {
      // pointer to pointer not supported
      assert(!(property & (long) kIsPointer));
      property |= kIsPointer;
      if(property & (long) kIsConstant) {
        property &= ~((long) kIsConstant);
        property |= kIsConstPointer;
      }
      return byName(name.substr(0, name.size() - 1), property);
    }

    TypeMap::const_iterator it = typeMap.find(name);
    if (it != typeMap.end()) {
      if(property == 0L) {
        return it->second;
      }
      TypeWithDict twd = it->second;
      twd.property_ = property;
      return twd;
    }

    // Handle classes
    TClass* theClass = TClass::GetClass(name.c_str());
    if (theClass != nullptr && theClass->GetTypeInfo() != nullptr) {
      return TypeWithDict(theClass, property);
    }

    // Handle enums
    TEnum* theEnum = TEnum::GetEnum(name.c_str(), TEnum::kAutoload);
    if(theEnum) {
      return TypeWithDict(theEnum, name, property);
    }

    // Handle built-ins
    TDataType* theDataType = gROOT->GetType(name.c_str());
    if(theDataType) {
      switch(theDataType->GetType()) {
      case kUInt_t:
        return TypeWithDict(typeid(unsigned int), property);
      case kInt_t:
        return TypeWithDict(typeid(int), property);
      case kULong_t:
        return TypeWithDict(typeid(unsigned long), property);
      case kLong_t:
        return TypeWithDict(typeid(long), property);
      case kULong64_t:
        return TypeWithDict(typeid(unsigned long long), property);
      case kLong64_t:
        return TypeWithDict(typeid(long long), property);
      case kUShort_t:
        return TypeWithDict(typeid(unsigned short), property);
      case kShort_t:
        return TypeWithDict(typeid(short), property);
      case kUChar_t:
        return TypeWithDict(typeid(unsigned char), property);
      case kChar_t:
        return TypeWithDict(typeid(char), property);
      case kBool_t:
        return TypeWithDict(typeid(bool), property);
      case kFloat_t:
        return TypeWithDict(typeid(float), property);
      case kFloat16_t:
        return TypeWithDict(typeid(Float16_t), property);
      case kDouble_t:
        return TypeWithDict(typeid(double), property);
      case kDouble32_t:
        return TypeWithDict(typeid(Double32_t), property);
      case kCharStar:
        return TypeWithDict(typeid(char*), property);
      case kDataTypeAliasSignedChar_t:
        return TypeWithDict(typeid(signed char), property);
      }
    }

    if(name == "void") {
      return TypeWithDict(typeid(void), property);
    }

    // For a reason not understood, TClass::GetClass sometimes cannot find std::type_info
    // by name.  This simple workaround bypasses the problem.
    // The problem really should be debugged.  (testORA12)
    if(name == "std::type_info") {
      return TypeWithDict(typeid(std::type_info), property);
    }

    // std::cerr << "DEBUG BY NAME: " << name << std::endl;
    TType* type = gInterpreter->Type_Factory(name);
    if (!gInterpreter->Type_IsValid(type)) {
      typeMap.insert(std::make_pair(name, TypeWithDict()));
      return TypeWithDict();
    }
    typeMap.insert(std::make_pair(name, TypeWithDict(type, 0L)));
    return TypeWithDict(type, property);
  }

  TypeWithDict::TypeWithDict() :
    ti_(&typeid(TypeWithDict::invalidType)),
    type_(nullptr),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
    property_(0L) {
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& rhs) :
    ti_(rhs.ti_),
    type_(rhs.type_),
    class_(rhs.class_),
    enum_(rhs.enum_),
    dataType_(rhs.dataType_),
    arrayDimensions_(rhs.arrayDimensions_),
    property_(rhs.property_) {
  }

  TypeWithDict&
  TypeWithDict::stripConstRef() {
    if(isPointer()) {
      property_ &= ~((long) kIsReference | (long) kIsConstPointer);
    } else {
      property_ &= ~((long) kIsConstant | (long) kIsReference);
    }
    return *this;
  }

  TypeWithDict&
  TypeWithDict::operator=(TypeWithDict const& rhs) {
    if (this != &rhs) {
      ti_ = rhs.ti_;
      type_ = rhs.type_;
      class_ = rhs.class_;
      enum_ = rhs.enum_;
      dataType_ = rhs.dataType_;
      arrayDimensions_ = rhs.arrayDimensions_;
      property_ = rhs.property_;
    }
    return *this;
  }

  TypeWithDict::TypeWithDict(std::type_info const& ti) : TypeWithDict(ti, 0L) {
  }

  TypeWithDict::TypeWithDict(std::type_info const& ti, long property /*= 0L*/) :
    ti_(&ti),
    type_(nullptr),
    class_(TClass::GetClass(ti)),
    enum_(nullptr),
    dataType_(TDataType::GetDataType(TDataType::GetType(ti))),
    arrayDimensions_(nullptr),
    property_(property) {

    if(class_ != nullptr) {
      return;
    }

    // Handle pointers
    // Must be done before dataType_ is checked, because dataType_ will be filled for char*
    if (TypeID(*ti_).className().back() == '*') {
      *this = TypeWithDict::byName(TypeID(*ti_).className());
      return;
    }

    if(dataType_ != nullptr) {
      return;
    }

    enum_ = TEnum::GetEnum(ti, TEnum::kAutoload);
    if(enum_ != nullptr) {
      return;
    }

    if(ti == typeid(void)) {
      // For some reason, "void" has a data type if accessed by name, but not by type_info.
      dataType_ = gROOT->GetType("void");
      return;
    }

    // std::cerr << "DEBUG BY TI: " << name() << std::endl;

    type_ = gInterpreter->Type_Factory(ti);
    if (!gInterpreter->Type_IsValid(type_)) {
      throwTypeException("TypeWithDict(TType*, property): ", name());
    }
  }

  TypeWithDict::TypeWithDict(TClass* cl, long property /*= 0L*/) :
    ti_(cl->GetTypeInfo()),
    type_(nullptr),
    class_(cl),
    enum_(nullptr),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
    property_(property) {
  }

  TypeWithDict::TypeWithDict(TEnum* enm, std::string const& name, long property /*= 0L*/) :
    ti_(&typeid(TypeWithDict::dummyType)),
    type_(nullptr),
    class_(nullptr),
    enum_(enm),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
    property_(property) {
  }

  TypeWithDict::TypeWithDict(TMethodArg* arg) : TypeWithDict(arg, 0L) {
  }

  TypeWithDict::TypeWithDict(TMethodArg* arg, long property) :
    TypeWithDict(byName(arg->GetTypeName(), arg->Property() | property)) {
  }

  TypeWithDict::TypeWithDict(TType* ttype, long property /*= 0L*/) :
    ti_(&typeid(invalidType)),
    type_(ttype),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
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
    // if(dataType_ != nullptr) {
    //  std::cerr << "DEBUG BY TTYPE FUNDAMENTAL: " << name() << std::endl;
    // }
    if (!gInterpreter->Type_IsFundamental(ttype) &&
        !gInterpreter->Type_IsArray(ttype) &&
        !gInterpreter->Type_IsPointer(ttype) &&
        !gInterpreter->Type_IsEnum(ttype)) {
      // Must be a class, struct, or union.
      class_ = TClass::GetClass(*ti_);
      // if(class_ != nullptr) {
      //  std::cerr << "DEBUG BY TTYPE CLASS: " << name() << std::endl;
      //  return;
      // }
    }
    if (gInterpreter->Type_IsEnum(ttype)) {
      enum_ = TEnum::GetEnum(*ti_, TEnum::kAutoload);
      // if(enum_ != nullptr) {
      //  std::cerr << "DEBUG BY TTYPE ENUM: " << name() << std::endl;
      // }
    }
  }

  TypeWithDict::operator bool() const {
    if (*ti_ == typeid(invalidType)) {
      return false;
    }
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr) {
      return true;
    }
    assert(type_ != nullptr);
    return gInterpreter->Type_Bool(type_);
  }

  std::type_info const&
  TypeWithDict::typeInfo() const {
    if(*ti_ == typeid(dummyType) || isPointer() || isArray()) {
      // No accurate type_info
      assert(qualifiedName().c_str() == nullptr);
    }
    return *ti_;
  }

  TClass*
  TypeWithDict::getClass() const {
    if(isPointer() || isArray()) {
      return nullptr;
    }
    return class_;
  }

  TEnum*
  TypeWithDict::getEnum() const {
    if(isPointer() || isArray()) {
      return nullptr;
    }
    return enum_;
  }

  TDataType*
  TypeWithDict::getDataType() const {
    if(isPointer() || isArray()) {
      return nullptr;
    }
    return dataType_;
  }

  long
  TypeWithDict::getProperty() const {
    return property_;
  }

  bool
  TypeWithDict::isClass() const {
    // Note: This really means is class, struct, or union.
    return class_ != nullptr && !isPointer() && !isArray();
  }

  bool
  TypeWithDict::isConst() const {
    return (property_ & (long) kIsConstant);
  }

  bool
  TypeWithDict::isArray() const {
    if (*ti_ == typeid(invalidType)) {
      return false;
    }
    return (type_ != nullptr && name().back() == ']');
  }

  bool
  TypeWithDict::isEnum() const {
    return enum_ != nullptr && !isPointer() && !isArray();
  }

  bool
  TypeWithDict::isFundamental() const {
    return dataType_ != nullptr && !isPointer() && !isArray();
  }

  bool
  TypeWithDict::isPointer() const {
    return (property_ & (long) kIsPointer);
  }

  bool
  TypeWithDict::isReference() const {
    return (property_ & (long) kIsReference);
  }

  bool
  TypeWithDict::isTemplateInstance() const {
    return (isClass() && name().back() == '>');
  }

  bool
  TypeWithDict::isTypedef() const {
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr || *ti_ == typeid(invalidType)) {
      return false;
    }
    assert(type_ != nullptr);
    return gInterpreter->Type_IsTypedef(type_);
  }

  bool
  TypeWithDict::isVirtual() const {
    return isClass() && (class_->ClassProperty() & (long) kClassHasVirtual);
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
    if (isConst() && !isPointer()) {
      qname = "const " + qname;
    } else if (property_ & kIsConstPointer) {
      qname += " const";
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
    if (*ti_ == typeid(invalidType)) {
      return std::string();
    }
    std::string suffix;
    std::string prefix;
    if(isPointer()) {
      suffix += "*";
      if(isConst()) {
        prefix += "const ";
      }
    }
    if(enum_ != nullptr) {
      if(enum_->GetClass()) {
         return prefix + std::string(enum_->GetClass()->GetName()) + "::" + enum_->GetName() + suffix;
      }
      return prefix + enum_->GetName() + suffix;
    }
    return prefix + TypeID(*ti_).className() + suffix;
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
    return name();
  }

  std::string
  TypeWithDict::friendlyClassName() const {
    return friendlyname::friendlyName(name());
  }

  size_t
  TypeWithDict::size() const {
    if(isPointer()) {
      return sizeof(void*);
    }
    if(class_ != nullptr) {
      return class_->GetClassSize();
    }
    if(dataType_ != nullptr) {
      return dataType_->Size();
    }
    if(enum_ != nullptr) {
      return sizeof(int);
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
    if (isClass()) {
      return class_->GetListOfDataMembers()->GetSize();
    }
    if (isEnum()) {
      return enum_->GetConstants()->GetSize();
    }
    return 0;
  }

  size_t
  TypeWithDict::functionMemberSize() const {
    if (isClass()) {
      return class_->GetListOfMethods()->GetSize();
    }
    return 0;
  }

  void const*
  TypeWithDict::pointerToBaseType(void const* ptr, TypeWithDict const& derivedType) const {
    if(!isClass()) {
      return ptr;
    }
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
    if(!isClass()) {
      return ptr;
    }
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
    if (isClass()) {
      TDataMember* dataMember = class_->GetDataMember(member.c_str());
      if(dataMember == nullptr) {
        // Look for indirect data members
        TRealData* realDataMember = class_->GetRealData(member.c_str());
        if(realDataMember != nullptr) {
          dataMember = realDataMember->GetDataMember();
        }
      }
      return MemberWithDict(dataMember);
    }
    if (isEnum()) {
      TClass* cl = enum_->GetClass();
      return MemberWithDict(cl->GetDataMember(member.c_str()));
    }
    return MemberWithDict();
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& member) const {
    if (!isClass()) {
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
    if (!isClass()) {
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
    if (*ti_ == typeid(invalidType)) {
      return TypeWithDict();
    }
    if(!isClass() && !isFundamental()) {
      return *this;
    }
    return TypeWithDict(*ti_);
  }

  TypeWithDict
  TypeWithDict::toType() const {
    if (*ti_ == typeid(invalidType)) {
      return TypeWithDict();
    }
    if(isReference()) {
      TypeWithDict newType = *this;
      newType.property_ &= ~((long) kIsReference);
      return newType;
    }
    if(isPointer()) {
      TypeWithDict newType = *this;
      newType.property_ &= ~((long) kIsPointer | (long) kIsConstPointer);
      return newType;
    }
    if(isArray()) {
      assert(type_ != nullptr);
      TType* ty = gInterpreter->Type_ToType(type_);
      if (ty == nullptr) {
        return *this;
      }
      std::type_info const* ti = gInterpreter->Type_TypeInfo(ty);
      if (ti == nullptr) {
        return *this;
      }
      return TypeWithDict(*ti);
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
    if(!isClass()) {
      return TypeWithDict();
    }
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
    if(!isClass()) {
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
    if(!isClass()) {
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
    if(!isClass()) {
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
    if (!isEnum()) {
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
    if (isClass()) {
      return ObjectWithDict(*this, class_->New());
    }
    return ObjectWithDict(*this, new char[size()]);
  }

  void
  TypeWithDict::destruct(void* address, bool dealloc) const {
    if (isClass()) {
      class_->Destructor(address, !dealloc);
      return;
    }
    if (dealloc) {
      delete[] reinterpret_cast<char*>(address);
    }
  }

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
    return a.name() == b.name();
  }

  bool
  operator==(TypeWithDict const& a, std::type_info const& b) {
    if(*a.ti_ == typeid(TypeWithDict::dummyType) || a.isPointer() || a.isArray()) {
      // No accurate type_info
      return a.name() == TypeID(b).className();
    }
    return *a.ti_ == b;
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
    class_(type.getClass()) {
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
    class_(type.getClass()) {
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
    class_(type.getClass()) {
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
