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

  namespace {
    typedef tbb::concurrent_unordered_map<std::string, TypeWithDict> Map;
    Map typeMap;
    typedef tbb::concurrent_unordered_map<std::string, FunctionWithDict> FunctionMap;
    FunctionMap functionMap;
  }
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
    auto const& item = typeMap.find(name);
    if(item != typeMap.end()) {
       return item->second;
    }
    TypeWithDict theType = TypeWithDict::byName(name, 0L);
    typeMap.insert(std::make_pair(name, theType));
    return theType;
  }

  TypeWithDict
  TypeWithDict::byName(std::string const& name, long property) {
    // This is a private static function.

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
      // C-style array of pointers is not supported
      assert(!(property & (long) kIsArray));
      property |= kIsPointer;
      if(property & (long) kIsConstant) {
        property &= ~((long) kIsConstant);
        property |= kIsConstPointer;
      }
      return byName(name.substr(0, name.size() - 1), property);
    }

    // Handle C-style arrays
    if(name.back() == ']') {
      // pointer to array not supported
      assert(!(property & (long) kIsPointer));
      // Protect against the remote possibility of '[' nested in a class type
      size_t begin = name.find_last_of("<>:,()");
      if(begin == std::string::npos) {
        begin = 0;
      } else {
        ++begin;
      }
      size_t first = name.find('[', begin);
      assert(first != std::string::npos);
      assert(first != 0);
      TypeWithDict ret = TypeWithDict::byName(name.substr(0, first), property);
      ret.property_ |= kIsArray;
      ret.arrayDimensions_ = value_ptr<std::vector<size_t> >(new std::vector<size_t>);
      std::string const dimensions = name.substr(first);
      char const* s = dimensions.c_str();
      while(1) {
        size_t x = 0;
        int count = sscanf(s, "[%lu]", &x);
        assert(count == 1);
        ret.arrayDimensions_->push_back(x);
        ++s;
        while(*s != '\0' && *s != '[') {
          ++s;
        }
        if(*s == '\0') {
          break;
        }
      }
      return ret;
    }

    // Handle classes
    TClass* theClass = TClass::GetClass(name.c_str());
    if (theClass != nullptr && theClass->GetTypeInfo() != nullptr) {
      return TypeWithDict(theClass, property);
    }

    // Handle enums
    TEnum* theEnum = TEnum::GetEnum(name.c_str(), TEnum::kAutoload);
    if(theEnum) {
      return TypeWithDict(theEnum, property);
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

    // For a reason not understood, TClass::GetClass sometimes cannot find std::vector<T>::value_type
    // or std::map<X,T>::value_type when T is a nested class.
    // This workaround bypasses the problem. The problem really should be debugged.
    if(stripNamespace(name) == "value_type") {
      size_t begin = name.find('<');
      size_t end = name.rfind('>');
      if(begin != std::string::npos && end != std::string::npos && end > ++begin) {
        size_t amap = name.find("map");
        if(amap != std::string::npos && amap < begin) {
           ++end;
           return TypeWithDict::byName(std::string("std::pair<const ") + name.substr(begin, end - begin), property);
        }
        return TypeWithDict::byName(name.substr(begin, end - begin), property);
      }
    }
    //std::cerr << "DEBUG BY NAME: " << name << std::endl;
    return TypeWithDict();
  }

  TypeWithDict::TypeWithDict() :
    ti_(&typeid(TypeWithDict::invalidType)),
    class_(nullptr),
    enum_(nullptr),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
    property_(0L) {
  }

  TypeWithDict::TypeWithDict(TypeWithDict const& rhs) :
    ti_(rhs.ti_),
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
    class_(TClass::GetClass(ti)),
    enum_(nullptr),
    dataType_(TDataType::GetDataType(TDataType::GetType(ti))),
    arrayDimensions_(nullptr),
    property_(property) {

    if(class_ != nullptr) {
      return;
    }

    // Handle pointers and arrays
    // Must be done before dataType_ is checked, because dataType_ will be filled for char*
    char lastChar = TypeID(*ti_).className().back();
    if (lastChar == '*' || lastChar == ']') {
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

     throwTypeException("TypeWithDict): ", name());
  }

  TypeWithDict::TypeWithDict(TClass *cl) : TypeWithDict(cl, 0L) {
  }

  TypeWithDict::TypeWithDict(TClass* cl, long property) :
    ti_(cl->GetTypeInfo()),
    class_(cl),
    enum_(nullptr),
    dataType_(nullptr),
    arrayDimensions_(nullptr),
    property_(property) {
    if(ti_ == nullptr) {
      ti_ = &typeid(TypeWithDict::invalidType);
      class_ = nullptr;
      property_ = 0L;
    }
  }

  TypeWithDict::TypeWithDict(TEnum *enm) : TypeWithDict(enm, 0L) {
  }

  TypeWithDict::TypeWithDict(TEnum* enm, long property) :
    ti_(&typeid(TypeWithDict::dummyType)),
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

  TypeWithDict::operator bool() const {
    if (*ti_ == typeid(invalidType)) {
      return false;
    }
    if (class_ != nullptr || dataType_ != nullptr || enum_ != nullptr) {
      return true;
    }
    return false;
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
    return (property_ & (long) kIsArray);
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
    return true;
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
    std::ostringstream out;
    if(isPointer() && isConst()) {
      out << "const ";
    }
    if(enum_ != nullptr) {
      if(enum_->GetClass()) {
         out << std::string(enum_->GetClass()->GetName());
         out <<  "::";
      }
      out << enum_->GetName();
    } else {
      out << TypeID(*ti_).className();
    }
    if(isPointer()) {
      out << '*';
    }
    if(isArray()) {
      for(size_t i = 0; i < arrayDimension(); ++i) {
        out << '[';
        out << std::dec << maximumIndex(i);
        out << ']';
      }
    }
    return out.str();
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
    size_t nBytes = 0;
    if(isPointer()) {
      nBytes = sizeof(void*);
    } else if(class_ != nullptr) {
      nBytes = class_->GetClassSize();
    } else if(dataType_ != nullptr) {
      nBytes = dataType_->Size();
    } else if(enum_ != nullptr) {
      nBytes = sizeof(int);
    }
    if(isArray()) {
      nBytes *= arrayLength();
    }
    return nBytes;
  }

  size_t
  TypeWithDict::arrayLength() const {
    assert(isArray());
    size_t theLength = 1;
    for(size_t i = 0; i < arrayDimension(); ++i) {
      theLength *= maximumIndex(i);
    }
    return theLength;
  }

  size_t
  TypeWithDict::arrayDimension() const {
    assert(isArray());
    return arrayDimensions_->size();
  }

  size_t
  TypeWithDict::maximumIndex(size_t dim) const {
    assert(isArray());
    return (*arrayDimensions_)[dim];
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
  TypeWithDict::functionMemberByName(std::string const& functionName, std::string const& proto, bool isConst) const {
    if (!isClass()) {
      return FunctionWithDict();
    }
    std::string const& key = name() + '#' + functionName + '#' + proto;
    auto const& item = functionMap.find(key);
    if(item != functionMap.end()) {
       return item->second;
    }
    TMethod* meth = class_->GetMethodWithPrototype(functionName.c_str(), proto.c_str(), /*objectIsConst=*/isConst, /*mode=*/ROOT::kConversionMatch);
    if (meth == nullptr) {
      return FunctionWithDict();
    }
    FunctionWithDict theFunction = FunctionWithDict(meth);
    functionMap.insert(std::make_pair(key, theFunction));
    return theFunction;
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
      TypeWithDict newType = *this;
      size_t size = newType.arrayDimensions_->size();
      if(size == 1) {
        newType.property_ &= ~((long) kIsArray);
        value_ptr<std::vector<size_t> > emptyVec;
        newType.arrayDimensions_ = emptyVec;
      } else {
        std::vector<size_t>& dims = *newType.arrayDimensions_;
        for(size_t i = 0; i != size; ++i) {
          dims[i] = dims[i+1];
        }
        newType.arrayDimensions_->resize(size - 1);
      }
      return newType;
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
    return class_->GetListOfMethods(kFALSE)->GetSize();
  }

} // namespace edm
