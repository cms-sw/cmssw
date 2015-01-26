#include "CondCore/ORA/interface/Object.h"
#include "CondCore/ORA/interface/Exception.h"
#include "ClassUtils.h"
// externals
#include "FWCore/Utilities/interface/ObjectWithDict.h"

ora::Object::Object():
  m_ptr(0),
  m_type(){
}

ora::Object::Object( const void* ptr, const std::type_info& typeInfo ):
  m_ptr( const_cast<void*>(ptr) ){
  m_type = ClassUtils::lookupDictionary( typeInfo, true );
}

ora::Object::Object( const void* ptr, const edm::TypeWithDict& type ):
  m_ptr( const_cast<void*>(ptr) ),
  m_type( type ){  
}

ora::Object::Object( const void* ptr, const std::string& typeName ):
  m_ptr( const_cast<void*>(ptr) ),
  m_type(edm::TypeWithDict::byName( typeName )){
}

ora::Object::Object( const Object& rhs):
  m_ptr( rhs.m_ptr ),
  m_type( rhs.m_type ){
}

ora::Object::~Object(){
}

ora::Object& ora::Object::operator=( const Object& rhs){
  m_ptr = rhs.m_ptr;
  m_type = rhs.m_type;
  return *this;
}

bool ora::Object::operator==( const Object& rhs) const{
  if( m_ptr != rhs.m_ptr ) return false;
  if( m_type != rhs.m_type ) return false;
  return true;
}

bool ora::Object::operator!=( const Object& rhs) const{
  return !operator==( rhs );
}

void* ora::Object::address() const {
  return m_ptr;
}

const edm::TypeWithDict& ora::Object::type() const {
  return m_type;
}

std::string ora::Object::typeName() const {
  return m_type.cppName();
}

void* ora::Object::cast( const std::type_info& typeInfo ) const{
  edm::TypeWithDict castType = ClassUtils::lookupDictionary( typeInfo );
  if( ! m_type ){
    throwException( "Object input class has not been found in the dictionary.",
                    "Object::cast" );
    
  }
  return ClassUtils::upCast( m_type, m_ptr, castType );
}

boost::shared_ptr<void> ora::Object::makeShared( ) const {
  boost::shared_ptr<void> ret;
  if( m_ptr ) {
    ret = boost::shared_ptr<void>( m_ptr, RflxDeleter( m_type ) );
  }
  return ret;
}

void ora::Object::destruct() {
  if( ! m_type ){
    throwException( "Object input class has not been found in the dictionary.",
                    "Object::destruct" );
    
  }
  if( m_ptr ){
    m_type.destruct( m_ptr );
    m_ptr = 0;
  }
}

