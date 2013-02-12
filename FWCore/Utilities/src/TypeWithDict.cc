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
  bool
  check(edm::TypeWithDict const& me, TClass* cl, TDataType* dt) {
    if(!(cl != nullptr || dt != nullptr)) {
      return false;
    }
    if(dt != nullptr) {
      assert(dt->GetType() != kOther_t);
      assert(dt->GetType() != kCharStar);
    }
    if(cl != nullptr) {
      assert(cl->GetTypeInfo() != nullptr);
    }
    return true;
  }

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
  TypeWithDict::TypeWithDict() :
    typeInfo_(&typeid(void)),
    type_(),
    class_(nullptr),
    dataType_(nullptr) {
  }

  TypeWithDict::TypeWithDict(Reflex::Type const& type) :
    typeInfo_(&type.TypeInfo()),
    type_(type),
    class_(TClass::GetClass(*typeInfo_)),
    dataType_(TDataType::GetDataType(TDataType::GetType(*typeInfo_))),
    property_(0L) {
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
      if(type_.IsFundamental()) {
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
      if(type_.TypeInfo() == typeid(void)) {
         return;
      }
      if(class_ == nullptr) {
        if(isClass()) {
          // This is a class or struct with a Reflex dictionary and no CINT dictionary.
          // Enable Reflex now, and try again.
          ROOT::Cintex::Cintex::Enable();
          class_ = TClass::GetClass(*typeInfo_),
          assert(class_ != nullptr);
        }
      } else {
        assert(!isFundamental());
        property_ |= (Long_t)kIsClass;
      }
  }

  TypeWithDict::TypeWithDict(std::type_info const& t) :
    TypeWithDict{t, 0L} {
  }

  TypeWithDict::TypeWithDict(std::type_info const& t, Long_t property) :
    typeInfo_(&t),
    type_(Reflex::Type::ByTypeInfo(t), toReflex(property)),
    class_(TClass::GetClass(t)),
    dataType_(TDataType::GetDataType(TDataType::GetType(t))),
    property_(property) {
      if(type_.IsEnum()) {
        // Enumerations are a special case, which we still handle with Reflex:
        property_ |= (Long_t)kIsEnum;
        return;
      }
      if(class_ == nullptr) {
        assert(!isClass());
        property_ |= (Long_t)kIsFundamental;
      } else {
        assert(!isFundamental());
        property_ |= (Long_t)kIsClass;
      }
  }

  TypeWithDict::TypeWithDict(TClass* cl, Long_t property) :
    typeInfo_(cl->GetTypeInfo()),
    type_(Reflex::Type::ByTypeInfo(*cl->GetTypeInfo()), toReflex(property)),
    class_(cl),
    dataType_(nullptr),
    property_(property) {
      bool ok = check(*this, class_, dataType_);
      assert(ok);
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
      bool ok = check(*this, class_, dataType_);
      assert(ok);
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& type, Long_t property) :
    // Only modifies const and reference.
    typeInfo_(type.typeInfo_),
    type_(type.type_, toReflex(property|property_)),
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
    static TypeMap typeMap;
    if(typeMap.empty()) {
      typeMap.insert(std::make_pair(std::string("bool"), TypeWithDict(typeid(bool))));
      typeMap.insert(std::make_pair(std::string("char"), TypeWithDict(typeid(char))));
      typeMap.insert(std::make_pair(std::string("unsigned char"), TypeWithDict(typeid(unsigned char))));
      typeMap.insert(std::make_pair(std::string("short"), TypeWithDict(typeid(short))));
      typeMap.insert(std::make_pair(std::string("unsigned short"), TypeWithDict(typeid(unsigned short))));
      typeMap.insert(std::make_pair(std::string("int"), TypeWithDict(typeid(int))));
      typeMap.insert(std::make_pair(std::string("unsigned int"), TypeWithDict(typeid(unsigned int))));
      typeMap.insert(std::make_pair(std::string("long"), TypeWithDict(typeid(long))));
      typeMap.insert(std::make_pair(std::string("unsigned long"), TypeWithDict(typeid(unsigned long))));
      typeMap.insert(std::make_pair(std::string("long long"), TypeWithDict(typeid(int))));
      typeMap.insert(std::make_pair(std::string("unsigned long long"), TypeWithDict(typeid(int))));
      typeMap.insert(std::make_pair(std::string("float"), TypeWithDict(typeid(float))));
      typeMap.insert(std::make_pair(std::string("double"), TypeWithDict(typeid(double))));
      // typeMap.insert(std::make_pair(std::string("long double"), TypeWithDict(typeid(long double)))); // ROOT does not seem to know about long double
      typeMap.insert(std::make_pair(std::string("string"), TypeWithDict(typeid(std::string))));
    }
    
    std::string cintName = ROOT::Cintex::CintName(name);
    char last = *cintName.rbegin();
    if(last == '*') {
     cintName = cintName.substr(0, cintName.size() - 1);
    }

    TClass* cl = TClass::GetClass(cintName.c_str());
    if(cl != nullptr) {
      // it's a class
      std::type_info const* typeInfo = cl->GetTypeInfo();
      if(typeInfo == nullptr) {
        return TypeWithDict();
      }
      // it's a class with a dictionary
      return TypeWithDict(*typeInfo, property);
    }
    TypeMap::const_iterator it = typeMap.find(cintName);
    if(it == typeMap.end()) {
      // This is an enum, or a class without a CINT dictionary.  We need Reflex to handle it.
      Reflex::Type t = Reflex::Type::ByName(name);
      return(bool(t) ? TypeWithDict(t) : TypeWithDict());
    }
    // its a built-in type
    return TypeWithDict(it->second, property);
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
    return ((class_ != nullptr && class_->GetTypeInfo() != nullptr) || dataType_ != nullptr);
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
    }
    if(dealloc) {
      delete [] static_cast<char *>(address);
    }
  }

  void const*
  TypeWithDict::pointerToContainedType(void const* ptr, TypeWithDict const& containedType) const {
    // The const_cast below is needed because
    // Object's constructor requires a pointer to
    // non-const void, although the implementation does not, of
    // course, modify the object to which the pointer points.
    Reflex::Object obj(containedType.type_, const_cast<void*>(ptr));
    if (containedType.type_ == type_) return obj.Address();
    Reflex::Object cast = obj.CastObject(type_);
    return cast.Address(); // returns void*, after pointer adjustment
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

  TypeWithDict
  TypeWithDict::toType() const{
    return TypeWithDict(type_.ToType());
  }

  TypeWithDict
  TypeWithDict::templateArgumentAt(size_t index) const {
    std::string className = unscopedName();
    size_t begin = className.find('<');
    if(begin == std::string::npos) {
      return TypeWithDict();
    }
    size_t end = className.rfind('>');
    assert(end != std::string::npos);
    assert(begin < end);
    int depth = 0;
    size_t argCount = 0;
    for(size_t inx = begin + 1 ; inx < end; ++inx) {
      char c = className[inx];
      if(c == '<') {
        ++depth;
      } else if(c == '>') {
        --depth;
        assert(depth >= 0);
      } else if(depth == 0 && c == ',') {
        if(argCount < index) {
          begin = inx;
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

  TypeTemplateWithDict::TypeTemplateWithDict(TypeWithDict const& type) : typeTemplate_(type.type_.TemplateFamily()) {
  }

  TypeTemplateWithDict::TypeTemplateWithDict(Reflex::TypeTemplate const& typeTemplate) : typeTemplate_(typeTemplate) {
  }

  TypeTemplateWithDict
  TypeTemplateWithDict::byName(std::string const& templateName, int n) {
    Reflex::TypeTemplate t = Reflex::TypeTemplate::ByName(templateName, n);
    return(bool(t) ? TypeTemplateWithDict(t) : TypeTemplateWithDict());
  }

  std::string
  TypeTemplateWithDict::name(int mod) const {
    return typeTemplate_.Name(mod);
  }

  bool
  TypeTemplateWithDict::operator==(TypeTemplateWithDict const& other) const {
    return typeTemplate_ == other.typeTemplate_;
  }

  TypeTemplateWithDict::operator bool() const {
    return bool(typeTemplate_);
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
