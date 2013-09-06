#include "CondCore/CondDB/interface/Binary.h"
//
#include "CoralBase/Blob.h"
//
#include <cstring>

#include <openssl/sha.h>
#include <cstring>

conddb::Binary::Binary():
  m_data(){
}

conddb::Binary::Binary( const void* data, size_t size  ):
  m_data( new coral::Blob( size ) ){
  ::memcpy( m_data->startingAddress(), data, size );
}

conddb::Binary::Binary( const coral::Blob& data ):
  m_data( new coral::Blob(data.size()) ){
  ::memcpy( m_data->startingAddress(), data.startingAddress(), data.size() );
}

conddb::Binary::Binary( const Binary& rhs ):
  m_data( rhs.m_data ){
}

conddb::Binary& conddb::Binary::operator=( const Binary& rhs ){
  if( this != &rhs ) m_data = rhs.m_data;
  return *this;
}

const coral::Blob& conddb::Binary::get() const {
  return *m_data;
}

void conddb::Binary::copy( const std::string& source ){
  m_data.reset( new coral::Blob( source.size() ) );
  ::memcpy( m_data->startingAddress(), source.c_str(), source.size() );
}

const void* conddb::Binary::data() const {
  return m_data->startingAddress();
}
void* conddb::Binary::data(){
  return m_data->startingAddress();
}

size_t conddb::Binary::size() const {
  return m_data->size();
}

std::string conddb::Binary::hash() const {
  unsigned char hashData[20];                                                                                                                    
  SHA1(static_cast<const unsigned char *>(m_data->startingAddress()), m_data->size(), hashData );
  char tmp[20*2+1];
  // re-write bytes in hex
  for (unsigned int i = 0; i < 20; i++) {                                                                                                        
    ::sprintf(&tmp[i * 2], "%02x", hashData[i]);                                                                                                 
  }                                                                                                                                              
  tmp[20*2] = 0;                                                                                                                                 
  return tmp;                                                                                                                                    
}                                                                                                                                                



