#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "ClassUtils.h"
//
#include <cxxabi.h>
// externals
#include "Reflex/Object.h"

ora::RflxDeleter::RflxDeleter( const Reflex::Type& type ):
  m_type( type ){
}

ora::RflxDeleter::RflxDeleter( const RflxDeleter& rhs ):
  m_type( rhs.m_type ){
}

ora::RflxDeleter::~RflxDeleter(){
}

void ora::RflxDeleter::operator()( void* ptr ){
  m_type.Destruct( ptr );
}

void* ora::ClassUtils::upCast( const Reflex::Type& type,
                               void* ptr,
                               const Reflex::Type& castType ){
  void* ret = 0;
  if( type == castType ){
    ret = ptr;
  } else if( type.HasBase( castType )){
    Reflex::Object theObj( type, ptr );
    ret = theObj.CastObject( castType ).Address();
  }
  return ret;
}

bool ora::ClassUtils::isType( const Reflex::Type& type,
                              const Reflex::Type& baseType ){
  bool ret = false;
  if( type == baseType || type.HasBase( baseType )){
    ret = true;
  }
  return ret;
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

Reflex::Type ora::ClassUtils::lookupDictionary( const std::type_info& typeInfo, bool throwFlag ){
  Reflex::Type type = Reflex::Type::ByTypeInfo( typeInfo );
  if( typeInfo == typeid(std::string) ){
    // ugly, but no other way with Reflex...
    type = Reflex::Type::ByName("std::string");
  }
  if( !type && throwFlag ){
    throwException( "Class \""+demangledName(typeInfo)+"\" has not been found in the dictionary.",
                    "ClassUtils::lookupDictionary" );
  }
  return type;
}

Reflex::Type ora::ClassUtils::lookupDictionary( const std::string& className, bool throwFlag   ){
  Reflex::Type type = Reflex::Type::ByName( className );
  if( className == "std::basic_string<char>" ){
    // ugly, but no other way with Reflex...
    type = Reflex::Type::ByName("std::string");
  }
  if( !type && throwFlag ){
    throwException( "Class \""+className+"\" has not been found in the dictionary.",
                    "ClassUtils::lookupDictionary" );
  }
  return type;
}

void* ora::ClassUtils::constructObject( const Reflex::Type& typ ){
  void* ptr = 0;
  if( typ.Name(Reflex::SCOPED)=="std::string"){
    ptr = new std::string("");
  } else {
    ptr = typ.Construct().Address();
  }
  return ptr;
}

bool ora::ClassUtils::isTypeString(const Reflex::Type& typ){
  std::string name = typ.Name(Reflex::SCOPED|Reflex::FINAL);
  return ( name == "std::string" ||
           name == "std::basic_string<char>" );
}

bool ora::ClassUtils::isTypePrimitive(const Reflex::Type& typ){
  return ( typ.IsFundamental() || typ.IsEnum() || isTypeString( typ ) );
}

bool ora::ClassUtils::isTypeContainer(const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
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

bool ora::ClassUtils::isTypeKeyedContainer(const Reflex::Type& typ){
  Reflex::TypeTemplate tt = typ.TemplateFamily();
  if (! tt) {
    return false;
  } else {
    std::string contName = tt.Name(Reflex::SCOPED|Reflex::FINAL); 
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

bool ora::ClassUtils::isTypeNonKeyedContainer(const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
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
 
bool ora::ClassUtils::isTypeAssociativeContainer(const Reflex::Type& typ){
  Reflex::TypeTemplate tt = typ.TemplateFamily();
  if (! tt) {
    return false;
  } else {
    std::string contName = tt.Name(Reflex::SCOPED|Reflex::FINAL); 
    if(  contName == "std::map"            ||
         contName == "std::multimap"       ||
         contName == "__gnu_cxx::hash_map" ||
         contName == "__gnu_cxx::hash_multimap" ){
      return true;
    }
  }  
  return false;
}

bool ora::ClassUtils::isTypeNonAssociativeContainer(const Reflex::Type& typ){
  Reflex::TypeTemplate tt = typ.TemplateFamily();
  if (! tt) {
    return false;
  } else {
    std::string contName = tt.Name(Reflex::SCOPED|Reflex::FINAL); 
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


bool ora::ClassUtils::isTypeOraPointer( const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
    if(  contName == "ora::Ptr" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeOraReference( const Reflex::Type& typ){
  return typ.HasBase(Reflex::Type::ByTypeInfo(typeid(ora::Reference)));
}

bool ora::ClassUtils::isTypeUniqueReference( const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
    if(  contName == "ora::UniqueRef" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypePVector( const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
    if(  contName == "ora::PVector" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeQueryableVector( const Reflex::Type& typ){
  Reflex::TypeTemplate templ = typ.TemplateFamily();
  if (! templ) {
    return false;
  } else {
    std::string contName = templ.Name(Reflex::SCOPED|Reflex::FINAL); 
    if(  contName == "ora::QueryableVector" ){
       return true;
    }
  }
  return false;  
}

bool ora::ClassUtils::isTypeObject( const Reflex::Type& typ){
  Reflex::Type resType = ClassUtils::resolvedType( typ );
  if( isTypePrimitive( resType ) ) {
    //if ( resType.IsFundamental() || resType.IsEnum() || isTypeString(resType) ) {
    return false;
  } else {
    if( resType.IsArray() ) return false;
    if( isTypeContainer( resType ) ) return false;
    if( isTypeOraPointer( resType ) ) return false;
    if( isTypeUniqueReference( resType ) ) return false;
    if( isTypePVector( resType ) ) return false;
    if( isTypeQueryableVector( resType ) ) return false;
  }
  return true;
}

Reflex::Type ora::ClassUtils::containerValueType(const Reflex::Type& typ){
  Reflex::Type valueType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    Reflex::Type sti = typ.SubTypeAt(i);    
    if(sti.Name()=="value_type") {
      valueType = sti;
      break;
    }    
    i++;
  }
  return valueType;
}

Reflex::Type ora::ClassUtils::containerKeyType(const Reflex::Type& typ){
  Reflex::Type keyType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    Reflex::Type sti = typ.SubTypeAt(i);    
    if(sti.Name()=="key_type") {
      keyType = sti;
      break;
    }    
    i++;
  }
  return keyType;
}

Reflex::Type ora::ClassUtils::containerDataType(const Reflex::Type& typ){
  Reflex::Type dataType;
  // find the iterator return type as the member value_type of the containers  
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    Reflex::Type sti = typ.SubTypeAt(i);    
    if(sti.Name()=="mapped_type") {
      dataType = sti;
      break;
    }    
    i++;
  }
  return dataType;
}

Reflex::Type ora::ClassUtils::containerSubType(const Reflex::Type& typ, const std::string& subTypeName){
  Reflex::Type subType;
  size_t subTypeSize = typ.SubTypeSize();
  size_t i=0;
  while(i<subTypeSize){
    Reflex::Type sti = typ.SubTypeAt(i);
    if(sti.Name()==subTypeName) {
      subType = sti;
      break;
    }
    i++;
  }
  return resolvedType(subType);
}

Reflex::Type ora::ClassUtils::resolvedType(const Reflex::Type& typ){
  Reflex::Type resolvedType = typ;
  while(resolvedType.IsTypedef()){
    resolvedType = resolvedType.ToType();
  }
  return resolvedType;
}
