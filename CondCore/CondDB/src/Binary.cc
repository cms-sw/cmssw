#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include "CoralBase/Blob.h"
//
#include <cstring>

#include <openssl/sha.h>
#include <cstring>

cond::Binary::Binary():
  m_data( new coral::Blob(0) ){
}

cond::Binary::Binary( const void* data, size_t size  ):
  m_data( new coral::Blob( size ) ){
  ::memcpy( m_data->startingAddress(), data, size );
}

cond::Binary::Binary( const coral::Blob& data ):
  m_data( new coral::Blob(data.size()) ){
  ::memcpy( m_data->startingAddress(), data.startingAddress(), data.size() );
}

cond::Binary::Binary( const Binary& rhs ):
  m_data( rhs.m_data ),
  m_object( rhs.m_object ){
}

cond::Binary& cond::Binary::operator=( const Binary& rhs ){
  if( this != &rhs ) {
    m_data = rhs.m_data;
    m_object = rhs.m_object;
  }
  return *this;
}

const coral::Blob& cond::Binary::get() const {
  return *m_data;
}

void cond::Binary::copy( const std::string& source ){
  m_data.reset( new coral::Blob( source.size() ) );
  ::memcpy( m_data->startingAddress(), source.c_str(), source.size() );
  m_object = ora::Object();
}

const void* cond::Binary::data() const {
  if(!m_data.get()) throwException( "Binary data can't be accessed.","Binary::data");
  return m_data->startingAddress();
}
void* cond::Binary::data(){
  if(!m_data.get()) throwException( "Binary data can't be accessed.","Binary::data");
  return m_data->startingAddress();
}

size_t cond::Binary::size() const {
  if(!m_data.get()) throwException( "Binary data can't be accessed.","Binary::size");
  return m_data->size();
}
    
ora::Object cond::Binary::oraObject() const {
  return m_object;
}

void cond::Binary::fromOraObject( const ora::Object& object ){
  m_object = object;
  m_data.reset();
}


