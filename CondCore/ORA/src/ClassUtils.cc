#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "CondCore/ORA/interface/NamedRef.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "ClassUtils.h"
//
#include <typeinfo>
#include <cxxabi.h>
// externals
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "TROOT.h"
#include "TDictAttributeMap.h"

ora::RflxDeleter::RflxDeleter( const edm::TypeWithDict& type ):
  m_type( type ){
}

ora::RflxDeleter::RflxDeleter( const RflxDeleter& rhs ):
  m_type( rhs.m_type ){
}

ora::RflxDeleter::~RflxDeleter(){
}

void ora::RflxDeleter::operator()( void* ptr ){
  m_type.destruct( ptr );
}

void ora::ClassUtils::loadDictionary( const std::string& className ){
  static std::string const prefix("LCGReflex/");
  edmplugin::PluginCapabilities::get()->load(prefix + className);
}

void* ora::ClassUtils::upCast( const edm::TypeWithDict& type,
                               void* ptr,
                               const edm::TypeWithDict& castType ){
  void* ret = 0;
  if( type == castType ){
    ret = ptr;
  } else if( type.hasBase( castType )){
    ret = reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr) + type.getBaseClassOffset(castType));
  }
  return ret;
}

bool ora::ClassUtils::isType( const edm::TypeWithDict& type,
                              const edm::TypeWithDict& baseType ){
  bool ret = false;
  if( type == baseType || type.hasBase( baseType )){
    ret = true;
  }
  return ret;
}

static
void
replaceString(std::string& name, std::string const& from, std::string const& to) {
  // from must not be a substring of to.
  std::string::size_type length = from.size();
  std::string::size_type pos = 0;
  while((pos = name.find(from, pos)) != std::string::npos) {
     name.replace(pos, length, to);
  }
}

static
bool
compareWithSuffixes(std::string const& name, std::string const& other) {
  size_t pos = name.find('<');
  if(pos == std::string::npos) {
    // Not a template instance
    return false;
  }
  size_t nameSize = name.size();
  if(other.size() >= nameSize) {
    // Can't be a case with ull qualifiers no name.
    return false;
  }
  if(name.substr(0, pos) != other.substr(0, pos)) {
    return false;
  }
  int nesting = 0;
  size_t j = pos;
  for(size_t i = pos; i < nameSize; ++i) {
    char c = name[i];
    if (c == '<') ++nesting;
    if(c != other[j]) {
      if(c != 'u' && c != 'l') return false;
      if(nesting == 0) return false;
      if(!isdigit(name[i-1])) return false;
      for(size_t n = i-2; n >= pos; --n) {
        char q = name[n];
        if(q == '<' || q == ',') break;  
        if(!isdigit(q)) return false;
      }
      for(size_t n = i+1; n < nameSize; ++n) {
        char q = name[n];
        if(q == '>' || q == ',') break;  
        if(q != 'l') return false;
        ++i;
      }
      continue;
    }
    if (c == '>') --nesting;
    ++j;
  }
  return true;
}

bool ora::ClassUtils::checkMappedType( const edm::TypeWithDict& type, 
				                       const std::string& mappedTypeName ){
  if( isTypeString( type ) ){
    return (mappedTypeName=="std::basic_string<char>" || mappedTypeName=="basic_string<char>" || mappedTypeName=="std::string" || mappedTypeName=="string");
  } else if ( type.isEnum() ){
    return mappedTypeName=="int";
  } else if ( isTypeOraVector( type ) ){
    return isTypeNameOraVector( mappedTypeName );
  } else if ( type.cppName() == mappedTypeName ) {
    return true;
  }
  // ROOT 6 uses these typedefs in names.
  std::string typeName(mappedTypeName); 
  replaceString(typeName, "std::basic_string<char> ", "std::string");
  replaceString(typeName, "std::basic_string<char>", "std::string");
  if ( type.cppName() == typeName ) {
    return true;
  }
  // Ignore u and l qualifiers off of integer template parameters for comparison
  return ( compareWithSuffixes( type.cppName(), typeName ) );
}

bool ora::ClassUtils::findBaseType( edm::TypeWithDict& type, edm::TypeWithDict& baseType, size_t& func ){
  bool found = false;
  if ( ! type.hasBase(baseType) ) {
      return found; // no inheritance, nothing to do
  } else {
      func = type.getBaseClassOffset(baseType);
      found = true;
  }
/*-ap old code        
  for ( unsigned int i=0;i<type.BaseSize() && !found; i++){
     edm::BaseWithDict base = type.BaseAt(i);
     edm::TypeWithDict bt = resolvedType( base.ToType() );
     if( bt == baseType ){
       func = base.OffsetFP();
       found = true;
     } else {
       found = findBaseType( bt, baseType, func );
     }
  }
*/
  return found;
}

std::string ora::ClassUtils::demangledName( const std::type_info& typeInfo ){
  int status = 0;
  std::string ret("");
  char* realname = abi::__cxa_demangle( typeInfo.name(), 0, 0, &status);
  if( status == 0 && realname ){
    ret  = realname;
    free(realname);
  }
  return ret;
}

edm::TypeWithDict ora::ClassUtils::lookupDictionary( const std::type_info& typeInfo, bool throwFlag ){
  edm::TypeWithDict type ( typeInfo );
  if( typeInfo == typeid(std::string) ){
    // ugly, but no other way with Reflex...
    type = edm::TypeWithDict::byName("std::string");
  }
  if( !type ){
    loadDictionary( demangledName(typeInfo) );
    edm::TypeWithDict type1 ( typeInfo );
    type = type1;
  }
  if( !type && throwFlag ){
    throwException( "Class \""+demangledName(typeInfo)+"\" has not been found in the dictionary.",
                    "ClassUtils::lookupDictionary" );
  }
  return type;
}

edm::TypeWithDict ora::ClassUtils::lookupDictionary( const std::string& className, bool throwFlag   ){
  edm::TypeWithDict type = edm::TypeWithDict::byName( className );
  if( className == "std::basic_string<char>" ){
    // ugly, but no other way with Reflex...
    type = edm::TypeWithDict::byName("std::string");
  }
  if( !type ){
    loadDictionary( className );
    type = edm::TypeWithDict::byName( className );
  }
  if( !type && throwFlag ){
    throwException( "Class \""+className+"\" has not been found in the dictionary.",
                    "ClassUtils::lookupDictionary" );
  }
  return type;
}

void* ora::ClassUtils::constructObject( const edm::TypeWithDict& typ ){
  void* ptr = 0;
  if( typ.cppName()=="std::string"){
    ptr = new std::string("");
  } else {
    ptr = typ.construct().address();
  }
  return ptr;
}

bool ora::ClassUtils::isTypeString(const edm::TypeWithDict& typ){
  std::string name = typ.cppName();
  return ( name == "std::string" ||
           name == "std::basic_string<char>" );
}

bool ora::ClassUtils::isTypePrimitive(const edm::TypeWithDict& typ){
  return ( typ.isFundamental() || typ.isEnum() || isTypeString( typ ) );
}

bool ora::ClassUtils::isTypeContainer(const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "std::vector"              ||
         contName == "std::list"                ||
         contName == "std::deque"               ||
         contName == "std::stack"               ||
         contName == "std::set"                 ||
         contName == "std::multiset"            ||
         contName == "__gnu_cxx::hash_set"      ||
         contName == "__gnu_cxx::hash_multiset" ||
         contName == "std::map"                 ||
         contName == "std::multimap"            ||
         contName == "__gnu_cxx::hash_map"      ||
         contName == "__gnu_cxx::hash_multimap" ||
         contName == "ora::PVector"            ||
         contName == "ora::QueryableVector" ){
       return true;
    }
  }
  return false;
}

bool ora::ClassUtils::isTypeKeyedContainer(const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "std::map"                 ||
         contName == "std::multimap"            ||
         contName == "std::set"                 ||
         contName == "std::multiset"            ||
         contName == "__gnu_cxx::hash_set"      ||
         contName == "__gnu_cxx::hash_multiset" ||
         contName == "__gnu_cxx::hash_map"      ||
         contName == "__gnu_cxx::hash_multimap" ){
      return true;
    }
  }  
  return false;
}

bool ora::ClassUtils::isTypeNonKeyedContainer(const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "std::vector"              ||
         contName == "std::list"                ||
         contName == "std::deque"               ||
         contName == "std::stack"               ||
         contName == "ora::PVector"            ||
         contName == "ora::QueryableVector" ){
       return true;
    }
  }
  return false;
}
 
bool ora::ClassUtils::isTypeAssociativeContainer(const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "std::map"            ||
         contName == "std::multimap"       ||
         contName == "__gnu_cxx::hash_map" ||
         contName == "__gnu_cxx::hash_multimap" ){
      return true;
    }
  }  
  return false;
}

bool ora::ClassUtils::isTypeNonAssociativeContainer(const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "std::vector"              ||
         contName == "std::list"                ||
         contName == "std::deque"               ||
         contName == "std::stack"               ||
         contName == "std::set"                 ||
         contName == "std::multiset"            ||
         contName == "__gnu_cxx::hash_set"      ||
         contName == "__gnu_cxx::hash_multiset" ||
         contName == "ora::PVector"            ||
         contName == "ora::QueryableVector"){
       return true;
    }
  }
  return false;
}


bool ora::ClassUtils::isTypeOraPointer( const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "ora::Ptr" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeOraReference( const edm::TypeWithDict& typ){
  return typ.hasBase( edm::TypeWithDict(typeid(ora::Reference)) );
}

bool ora::ClassUtils::isTypeNamedReference( const edm::TypeWithDict& typ){
  return typ.hasBase( edm::TypeWithDict(typeid(ora::NamedReference)) );
}

bool ora::ClassUtils::isTypeUniqueReference( const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "ora::UniqueRef" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypePVector( const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "ora::PVector" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeQueryableVector( const edm::TypeWithDict& typ){
  if ( ! typ.isTemplateInstance() ) {
    return false;
  } else {
    std::string contName = typ.templateName(); 
    if(  contName == "ora::QueryableVector" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeOraVector( const edm::TypeWithDict& typ){
  if( isTypePVector( typ ) || isTypeQueryableVector( typ ) ){
    return true;
  }
  return false;
}

bool ora::ClassUtils::isTypeNameOraVector( const std::string& typeName ){
  size_t idx = typeName.find('<');
  if( idx != std::string::npos ){
    std::string tname = typeName.substr( 0, idx );
    return (tname == "ora::PVector" || tname == "ora::QueryableVector" || tname == "pool::PVector" );
  }
  return false;
}

bool ora::ClassUtils::isTypeObject( const edm::TypeWithDict& typ){
  edm::TypeWithDict resType = ClassUtils::resolvedType( typ );
  if( isTypePrimitive( resType ) ) {
    //if ( resType.IsFundamental() || resType.IsEnum() || isTypeString(resType) ) {
    return false;
  } else {
    if( resType.isArray( )               ) return false;
    if( isTypeContainer( resType )       ) return false;
    if( isTypeOraPointer( resType )      ) return false;
    if( isTypeUniqueReference( resType ) ) return false;
    if( isTypeOraVector( resType )       ) return false;
  }
  return true;
}

edm::TypeWithDict ora::ClassUtils::containerValueType(const edm::TypeWithDict& typ){
    /*-ap old code
  edm::TypeWithDict valueType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    edm::TypeWithDict sti = typ.SubTypeAt(i);    
    if(sti.Name()=="value_type") {
      valueType = sti;
      break;
    }    
    i++;
  }
  return valueType;
    */
    return typ.nestedType("value_type");
}

edm::TypeWithDict ora::ClassUtils::containerKeyType(const edm::TypeWithDict& typ){
    /*-ap old code
  edm::TypeWithDict keyType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    edm::TypeWithDict sti = typ.SubTypeAt(i);    
    if(sti.Name()=="key_type") {
      keyType = sti;
      break;
    }    
    i++;
  }
  return keyType;
  */
  return typ.nestedType("key_type");
  
}

edm::TypeWithDict ora::ClassUtils::containerDataType(const edm::TypeWithDict& typ){
    /*-ap old code
  edm::TypeWithDict dataType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    edm::TypeWithDict sti = typ.SubTypeAt(i);    
    if(sti.Name()=="mapped_type") {
      dataType = sti;
      break;
    }    
    i++;
  }
  return dataType;
  */
  return typ.nestedType("mapped_type");
}

edm::TypeWithDict ora::ClassUtils::containerSubType(const edm::TypeWithDict& typ, const std::string& subTypeName){
    /*-ap old code
  edm::TypeWithDict subType;
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    edm::TypeWithDict sti = typ.SubTypeAt(i);
    if(sti.Name()==subTypeName) {
      subType = sti;
      break;
    }
    i++;
  }
  return resolvedType(subType);
  */
  return typ.nestedType(subTypeName);
}

edm::TypeWithDict ora::ClassUtils::resolvedType(const edm::TypeWithDict& typ){
  if (typ.isTypedef()){
    return typ.finalType().toType();
  }
  return typ;
}

size_t ora::ClassUtils::arrayLength( const edm::TypeWithDict& typ ) {
  size_t arraySize = typ.arrayLength();
  edm::TypeWithDict arrayType = typ.toType();
  if( arrayType.isArray() ) arraySize /= arrayType.arrayLength();
  return arraySize;
}

std::string ora::ClassUtils::getClassProperty( const std::string& propertyName, 
					       const edm::TypeWithDict& type ){
  std::string ret("");
  TClass* rclass = type.getClass();
  if( rclass ){
    TDictAttributeMap* classProps = rclass->GetAttributeMap();
    if( classProps && classProps->HasKey( propertyName.c_str() ) ){
      ret = classProps->GetPropertyAsString( propertyName.c_str() );
    }
  }
  return ret;
}

std::string ora::ClassUtils::getDataMemberProperty( const std::string& propertyName, 
						    const edm::MemberWithDict& dataMember ){
  std::string ret("");
  TClass* declaringClass = dataMember.declaringType().getClass();
  if( declaringClass ){
    auto dm = declaringClass->GetDataMember( dataMember.name().c_str() );
    if( dm ) {
      TDictAttributeMap* memberProps = dm->GetAttributeMap();
      if( memberProps && memberProps->HasKey( propertyName.c_str() ) ){
	ret = memberProps->GetPropertyAsString( propertyName.c_str() );
      }
    }
  }
  return ret;
}
