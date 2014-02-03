#include "CondCore/CondDB/interface/Binary.h"
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
  m_data( rhs.m_data ){
}

cond::Binary& cond::Binary::operator=( const Binary& rhs ){
  if( this != &rhs ) m_data = rhs.m_data;
  return *this;
}

const coral::Blob& cond::Binary::get() const {
  return *m_data;
}

void cond::Binary::copy( const std::string& source ){
  m_data.reset( new coral::Blob( source.size() ) );
  ::memcpy( m_data->startingAddress(), source.c_str(), source.size() );
}

const void* cond::Binary::data() const {
  return m_data->startingAddress();
}
void* cond::Binary::data(){
  return m_data->startingAddress();
}

size_t cond::Binary::size() const {
  return m_data->size();
}

std::string cond::Binary::hash() const {
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



