/*----------------------------------------------------------------------
, Long_t property
----------------------------------------------------------------------*/
#include <cassert>
#include <ostream>
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "TROOT.h"
#include "TClass.h"
#include "TDataType.h"
#include "TMethodArg.h"
#include "Cintex/Cintex.h"
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "boost/thread/tss.hpp"

namespace ROOT {
  namespace Cintex {
    std::string CintName(std::string const&);
  }
}

namespace {
  int
  toReflex(Long_t property) {
    int prop = 0;
    if(property&(Long_t)kIsConstant) {
      prop |= Reflex::CONST;
    }
    if(property&(Long_t)kIsReference) {
      prop |= Reflex::REFERENCE;
    }
    return prop;
  }
}

namespace edm {
  void
  TypeWithDict::setProperty() {
    if(type_.IsClass()) {
       property_ |= (Long_t)kIsClass;
    }
    if(type_.IsConst()) {
       property_ |= (Long_t)kIsConstant;
    }
    if(type_.IsEnum()) {
       property_ |= (Long_t)kIsEnum;
    }
    if(type_.IsTypedef()) {
       property_ |= (Long_t)kIsTypedef;
    }
    if(dataType_ != nullptr || *typeInfo_ == typeid(void)) {
       property_ |= (Long_t)kIsFundamental;
    }
    if(type_.IsPointer()) {
       property_ |= (Long_t)kIsPointer;
    }
    if(type_.IsReference()) {
       property_ |= (Long_t)kIsReference;
    }
    if(type_.IsStruct()) {
       property_ |= (Long_t)kIsStruct;
    }
    if(type_.IsVirtual()) {
       property_ |= (Long_t)kIsVirtual;
    }
  }

  TypeWithDict::TypeWithDict() :
    typeInfo_(&typeid(void)),
    type_(),
    class_(nullptr),
    dataType_(nullptr),
    property_(0L) {
  }

  TypeWithDict::TypeWithDict(Reflex::Type const& type) :
    typeInfo_(&type.TypeInfo()),
    type_(type),
    class_(nullptr),
    dataType_(TDataType::GetDataType(TDataType::GetType(*typeInfo_))),
    property_(0L) {
      setProperty();
      if(!isFundamental() && !isEnum()) { 
        class_ = TClass::GetClass(*typeInfo_);
      }
  }

  TypeWithDict::TypeWithDict(std::type_info const& t) :
    TypeWithDict{t, 0L} {
  }

  TypeWithDict::TypeWithDict(std::type_info const& t, Long_t property) :
    typeInfo_(&t),
    type_(Reflex::Type::ByTypeInfo(t), toReflex(property)),
    class_(nullptr),
    dataType_(TDataType::GetDataType(TDataType::GetType(t))),
    property_(property) {
      setProperty();
      if(!isFundamental() && !isEnum()) { 
        class_ = TClass::GetClass(*typeInfo_);
      }
  }

  TypeWithDict::TypeWithDict(TClass* cl, Long_t property) :
    typeInfo_(cl->GetTypeInfo()),
    type_(Reflex::Type::ByTypeInfo(*cl->GetTypeInfo()), toReflex(property)),
    class_(cl),
    dataType_(nullptr),
    property_(property) {
      property_ |= (Long_t)kIsClass;
  }

  TypeWithDict::TypeWithDict(TMethodArg* arg) :
    typeInfo_(nullptr),
    type_(),
    class_(nullptr),
    dataType_(nullptr),
    property_(arg->Property()) {
      TypeWithDict type(byName(arg->GetTypeName(), arg->Property()));
      *this = std::move(type);
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& type, Long_t property) :
    // Only modifies const and reference.
    typeInfo_(type.typeInfo_),
    type_(type.type_, toReflex(property|type.property_)),
    class_(type.class_),
    dataType_(type.dataType_),
    property_(type.property_) {
      if(property&(Long_t)kIsConstant) {
        property_ |= (Long_t)kIsConstant;
      } else {
        property_ &= ~(Long_t)kIsConstant;
      }
      if(property&(Long_t)kIsReference) {
        property_ |= (Long_t)kIsReference;
      } else {
        property_ &= ~(Long_t)kIsReference;
      }
  }

  void
  TypeWithDict::print(std::ostream& os) const {
    try {
      os << name();
    } catch (cms::Exception const& e) {
      os << typeInfo().name();
    }
  }

  TypeWithDict
  TypeWithDict::byName(std::string const& name) {
    return byName(name, 0L);
  }

  TypeWithDict
  TypeWithDict::byName(std::string const& name, Long_t property) {
    // static map for built in types
    typedef std::map<std::string, TypeWithDict> TypeMap;  
    static const TypeMap typeMap = {
      {std::string("bool"), TypeWithDict(typeid(bool))},
      {std::string("char"), TypeWithDict(typeid(char))},
      {std::string("unsigned char"), TypeWithDict(typeid(unsigned char))},
      {std::string("short"), TypeWithDict(typeid(short))},
      {std::string("unsigned short"), TypeWithDict(typeid(unsigned short))},
      {std::string("int"), TypeWithDict(typeid(int))},
      {std::string("unsigned int"), TypeWithDict(typeid(unsigned int))},
      {std::string("long"), TypeWithDict(typeid(long))},
      {std::string("unsigned long"), TypeWithDict(typeid(unsigned long))},
      {std::string("long long"), TypeWithDict(typeid(long long))},
      {std::string("unsigned long long"), TypeWithDict(typeid(unsigned long long))},
      {std::string("float"), TypeWithDict(typeid(float))},
      {std::string("double"), TypeWithDict(typeid(double))},
      // {std::string("long double"), TypeWithDict(typeid(long double))}, // ROOT does not seem to know about long double
      {std::string("string"), TypeWithDict(typeid(std::string))},
      {std::string("void"), TypeWithDict(typeid(void))}
    };
    
    std::string cintName = ROOT::Cintex::CintName(name);
    char last = *cintName.rbegin();
    if(last == '*') {
     cintName = cintName.substr(0, cintName.size() - 1);
    }

    TypeMap::const_iterator it = typeMap.find(cintName);
    if(it != typeMap.end()) {
      return TypeWithDict(it->second, property);
    }

    TClass* cl = TClass::GetClass(cintName.c_str());
    if(cl != nullptr) {
      // it's a class
      std::type_info const* typeInfo = cl->GetTypeInfo();
      if(typeInfo == nullptr) {
        return TypeWithDict();
      }
      // it's a class with a dictionary
      return TypeWithDict(cl, property);
    }
    // This is an enum, or a class without a CINT dictionary.  We need Reflex to handle it.
    Reflex::Type t = Reflex::Type::ByName(name);
    return(bool(t) ? TypeWithDict(t) : TypeWithDict());
  }

  std::string
  TypeWithDict::qualifiedName() const {
    std::string qname(name());
    if(isConst()) {
      qname = "const " + qname;
    }
    if(isReference()) {
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
    if(isPointer()) {
      return TypeID(typeInfo()).className() + '*';
    }
    return TypeID(typeInfo()).className();
  }

  std::string
  TypeWithDict::userClassName() const {
    return TypeID(typeInfo()).userClassName();
  }

  std::string
  TypeWithDict::friendlyClassName() const {
    return TypeID(typeInfo()).friendlyClassName();
  }

  bool
  TypeWithDict::hasDictionary() const {
    if(isEnum() || isTypedef() || isPointer()) {
      return bool(type_);
    }
    if(isFundamental()) {
      return true;
    }
    return ((class_ != nullptr && class_->GetTypeInfo() != nullptr));
  }

  ObjectWithDict
  TypeWithDict::construct() const {
    if(class_ != nullptr) {
      return ObjectWithDict(*this, class_->New());
    }
    return ObjectWithDict(*this, new char[size()]);
  }

  void
  TypeWithDict::destruct(void* address, bool dealloc) const {
    if(class_ != nullptr) {
      class_->Destructor(address, !dealloc);
    } else if(dealloc) {
      delete [] static_cast<char *>(address);
    }
  }

  void const*
  TypeWithDict::pointerToBaseType(void const* ptr, TypeWithDict const& derivedType) const {
    if(derivedType == *this) {
      return ptr;
    }
    int offset = derivedType.getBaseClassOffset(*this);
    if(offset < 0) return nullptr;
    return static_cast<char const*>(ptr) + offset;
  }

  void const*
  TypeWithDict::pointerToContainedType(void const* ptr, TypeWithDict const& derivedType) const {
    return pointerToBaseType(ptr, derivedType);
  }

  TypeWithDict::operator bool() const {
    return hasDictionary();
  }

  TypeWithDict
  TypeWithDict::nestedType(char const* nestedName) const {
    return byName(name() + "::" + nestedName);
  }

  MemberWithDict
  TypeWithDict::dataMemberByName(std::string const& member) const {
    if(class_ == nullptr) {
      if(isEnum()) {
        std::string fullName(name());
        size_t lastColon = fullName.find_last_of("::");
        assert(lastColon != std::string::npos && lastColon != 0U);
        std::string theName(fullName.substr(0, lastColon - 1));
        TClass* cl = TClass::GetClass(theName.c_str());
        assert(cl != nullptr);
        return MemberWithDict(cl->GetDataMember(member.c_str()));
      }
      return MemberWithDict();
    }
    return MemberWithDict(class_->GetDataMember(ROOT::Cintex::CintName(member).c_str()));
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& member) const {
    return FunctionWithDict(type_.FunctionMemberByName(member));
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& member, TypeWithDict const& signature, int mods, TypeMemberQuery memberQuery) const {
    return FunctionWithDict(type_.FunctionMemberByName(member, signature.type_, mods, static_cast<Reflex::EMEMBERQUERY>(memberQuery)));
  }

/*
  // Implementation in ROOT

  TypeWithDict::functionMemberByName(std::string const& member) const {
    if(class_ == nullptr) {
      return FunctionWithDict();
    }
    return FunctionWithDict(class_->GetMethodAny(ROOT::Cintex::CintName(member).c_str()));
  }

  FunctionWithDict
  TypeWithDict::functionMemberByName(std::string const& member, TypeWithDict const& signature, int mods, TypeMemberQuery memberQuery) const {
    if(class_ == nullptr) {
      return FunctionWithDict();
    }
    // This is wrong.
    return FunctionWithDict(class_->GetMethodAny(ROOT::Cintex::CintName(member).c_str()));
  }
*/

  bool
  TypeWithDict::isClass() const {
    return property_&((Long_t)kIsClass|(Long_t)kIsStruct);
  }

  bool
  TypeWithDict::isConst() const {
    return property_&(Long_t)kIsConstant;
  }

  bool
  TypeWithDict::isEnum() const {
    return property_&(Long_t)kIsEnum;
  }

  bool
  TypeWithDict::isFundamental() const {
    return property_ & (Long_t)kIsFundamental;
  }

  bool
  TypeWithDict::isPointer() const {
    return property_ & (Long_t)kIsPointer;
  }

  bool
  TypeWithDict::isReference() const {
    return property_ & (Long_t)kIsReference;
  }

  bool
  TypeWithDict::isTypedef() const {
    return property_ & (Long_t)kIsTypedef;
  }

  bool
  TypeWithDict::isTemplateInstance() const {
    return *name().rbegin() == '>';
  }

  bool
  TypeWithDict::isVirtual() const {
    return property_ & (Long_t)kIsVirtual;
  }

  TypeWithDict
  TypeWithDict::toType() const{
    return TypeWithDict(type_.ToType());
  }

  std::string
  TypeWithDict::templateName() const {
    if (!isTemplateInstance()) {
      return std::string();
    }
    std::string templateName = name();
    size_t begin = templateName.find('<');
    assert(begin != std::string::npos);
    size_t end = templateName.rfind('<');
    assert(end != std::string::npos);
    assert(begin <= end);
    if(begin < end) {
      int depth = 1;
      for(size_t inx = begin + 1 ; inx <= end; ++inx) {
        char c = templateName[inx];
        if(c == '<') {
          if(depth == 0) {
            begin = inx;
          }
          ++depth;
        } else if(c == '>') {
          --depth;
          assert(depth >= 0);
        }
      }
    }
    return templateName.substr(0, begin);
  }

  TypeWithDict
  TypeWithDict::templateArgumentAt(size_t index) const {
    std::string className = unscopedName();
    size_t begin = className.find('<');
    if(begin == std::string::npos) {
      return TypeWithDict();
    }
    ++begin;
    size_t end = className.rfind('>');
    assert(end != std::string::npos);
    assert(begin < end);
    int depth = 0;
    size_t argCount = 0;
    for(size_t inx = begin; inx < end; ++inx) {
      char c = className[inx];
      if(c == '<') {
        ++depth;
      } else if(c == '>') {
        --depth;
        assert(depth >= 0);
      } else if(depth == 0 && c == ',') {
        if(argCount < index) {
          begin = inx + 1;
          ++argCount;
        } else {
          end = inx;
          break;
        }
      }
    }
    assert(depth == 0);
    if(argCount < index) {
      return TypeWithDict();
    }
    return byName(className.substr(begin, end - begin));
  }

  std::type_info const&
  TypeWithDict::typeInfo() const {
    return *typeInfo_;
  }

  std::type_info const&
  TypeWithDict::id() const {
    return typeInfo();
  }

  int
  TypeWithDict::getBaseClassOffset(TypeWithDict const& baseClass) const {
    assert(class_ != nullptr);
    assert(baseClass.class_ != nullptr);
    int offset = class_->GetBaseClassOffset(baseClass.class_);
    return offset;
  }

  int
  TypeWithDict::stringToEnumValue(std::string const& enumMemberName) const {
    assert(isEnum());
    Reflex::Member member = type_.MemberByName(enumMemberName);
    if (!member) {
      std::ostringstream err;
      err<<"StringToEnumValue Failure trying to convert " << enumMemberName << " to int value";
      throw cms::Exception("ConversionError",err.str());
    }
    if (member.TypeOf().TypeInfo() != typeid(int)) {
      std::ostringstream err;
      err << "Type "<<  member.TypeOf().Name() << " is not Enum";
      throw cms::Exception("ConversionError",err.str());
    }
    return Reflex::Object_Cast<int>(member.Get());
  }

  size_t
  TypeWithDict::dataMemberSize() const {
    if(class_ == nullptr) {
      if(isEnum()) {
        return type_.DataMemberSize();
      }
      return 0U;
    }
    return class_->GetListOfDataMembers()->GetSize();
  }

  size_t
  TypeWithDict::functionMemberSize() const {
    if(class_ == nullptr) {
      return 0U;
    }
    return class_->GetListOfMethods()->GetSize();
  }

  size_t
  TypeWithDict::size() const {
    if(isEnum() || isTypedef() || isPointer()) {
      return type_.SizeOf();
    }
    if(class_ != nullptr) {
      return class_->Size();
    }
    assert(dataType_ != nullptr);
    return dataType_->Size();
  }

  bool
  operator==(TypeWithDict const& a, TypeWithDict const& b) {
    return a.typeInfo() == b.typeInfo();
  }

  std::ostream&
  operator<<(std::ostream& os, TypeWithDict const& id) {
    id.print(os);
    return os;
  }

  IterWithDict<TBaseClass>
  TypeBases::begin() const {
    if(class_ == nullptr) {
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
    if(class_ == nullptr) {
      return type_.BaseSize();
    }
    return class_->GetListOfBases()->GetSize();
  }

  IterWithDict<TDataMember>
  TypeDataMembers::begin() const {
    if(class_ == nullptr) {
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
    if(class_ == nullptr) {
      return type_.DataMemberSize();
    }
    return class_->GetListOfDataMembers()->GetSize();
  }

/*
  IterWithDict<TMethod>
  TypeFunctionMembers::begin() const {
    if(class_ == nullptr) {
      return IterWithDict<TMethod>();
    }
    return IterWithDict<TMethod>(class_->GetListOfMethods());
  }

  IterWithDict<TMethod>
  TypeFunctionMembers::end() const {
    return IterWithDict<TMethod>();
  }
*/

  Reflex::Member_Iterator
  TypeFunctionMembers::begin() const {
    return type_.FunctionMember_Begin();
  }

  Reflex::Member_Iterator
  TypeFunctionMembers::end() const {
    return type_.FunctionMember_End();
  }

/*
  size_t
  TypeFunctionMembers::size() const {
    if(class_ == nullptr) {
      return 0U;
    }
    return class_->GetListOfMethods()->GetSize();
  }
*/

  size_t
  TypeFunctionMembers::size() const {
    return type_.FunctionMemberSize();
  }

}
