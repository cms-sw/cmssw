#include "CondCore/RegressionTest/interface/RegressionTestPayload.h"
#include <sstream>

Data::Data():
  m_i(-1),
  m_s(""),
  m_a(){
}
  
Data::Data( int seed ):
  m_i(seed),
  m_s(""),
  m_a(){
  std::stringstream ss;
  ss << "Data_"<<seed;
  m_s = ss.str();
  for( int i=0;i<seed;i++){
    m_a.push_back(i);
  }
}
  
bool Data::operator ==(const Data& rhs) const {
  if( m_i != rhs.m_i ) return false;
  if( m_s != rhs.m_s ) return false;
  if( m_a != rhs.m_a ) return false;
  return true;
}
bool Data::operator !=(const Data& rhs) const {
  return !operator==( rhs );
}

RegressionTestPayload::RegressionTestPayload(): 
  PrimitivePayload(), 
  ArrayPayload(),
  m_i( -1 ),
  m_data0(),
  m_data1(){
}

RegressionTestPayload::RegressionTestPayload( int seed  ): 
  PrimitivePayload( seed ) , 
  ArrayPayload( seed ),
  m_i( seed ),
  m_data0( seed ),
  m_data1(seed ){
}
bool RegressionTestPayload::operator ==(const RegressionTestPayload& rhs) const {
  if( PrimitivePayload::operator!=(rhs) ) return false;
  if( ArrayPayload::operator!=(rhs) ) return false;
  if( m_i != rhs.m_i ) return false;
  if( m_data0 != rhs.m_data0 ) return false;
  if( m_data1 != rhs.m_data1 ) return false;
  return true;
}
  
bool RegressionTestPayload::operator !=(const RegressionTestPayload& rhs) const {
  return !operator==( rhs );
}
