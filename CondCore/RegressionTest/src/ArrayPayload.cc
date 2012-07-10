#include "CondCore/RegressionTest/interface/ArrayPayload.h"
#include <sstream>

Param::Param():
  p_i( -1 ),
  p_s(""){
}

Param::Param( int seed ):
  p_i( seed ),
  p_s(""){
  std::stringstream ss;
  ss << seed;
  p_s = ss.str();   
}

bool Param::operator ==(const Param& rhs) const {
  if( p_i != rhs.p_i ) return false;
  if( p_s != rhs.p_s ) return false;
  return true;  
}

bool Param::operator !=(const Param& rhs) const {
  return !operator==( rhs );
}

ArrayPayload::ArrayPayload():
  m_i(-1),
  m_p0(),
  m_p1(),
  m_p2(),
  m_vec0(),
  m_vec1(),
  m_map0(),
  m_map1(),
  m_list(),
  m_set(),
  m_bitset(),
  m_vec2(),
  m_map2(),
  m_vec3(){
}

ArrayPayload::ArrayPayload( int seed ):
  m_i( seed + 99 ),
  m_p0(),
  m_p1(),
  m_p2(),
  m_vec0(),
  m_vec1(),
  m_map0(),
  m_map1(),
  m_list(),
  m_set(),
  m_bitset(),
  m_vec2(),
  m_map2(),
  m_vec3(){
  for( size_t i=0;i<4; i++ ){
    m_ai0[i]= seed + i;
  }
  for( size_t i=0;i<112; i++ ){
    m_ai1[i]= seed + i;
  }
  for( size_t i=0;i<2; i++ ){
    for( size_t j=0;i<3; i++ ){
      m_ai2[j][i]= seed + i + j;
    }
  }
  for( size_t i=0;i<80; i++ ){
    for( size_t j=0;i<2; i++ ){
      m_ai3[j][i]= seed + i + j;
    }
  }
  for( size_t i=0;i<4; i++ ){
    std::stringstream ss;
    ss << (seed + i);
    m_as0[i]= ss.str();
  }
  for( size_t i=0;i<112; i++ ){
    std::stringstream ss;
    ss << (seed + i);
    m_as1[i]= ss.str();
  }
  for( int i=0;i<4; i++ ){
    m_ap0[i] = Param(i+seed);
  }
  for( int i=0;i<112; i++ ){
    m_ap1[i] = Param(i+seed);
  }
  m_p0.first = seed;
  m_p0.second = seed*10;
  m_p1.first = seed;
  std::stringstream ss;
  ss << (seed*10);
  m_p1.second = ss.str();
  m_p2.first = seed;
  m_p2.second = Param( seed );
  
  for( int i=0;i<seed;i++){
    m_vec0.push_back(i);
    std::stringstream ss0;
    ss0 << "vec1_"<<seed;
    m_vec1.push_back( ss0.str() );
    m_map0.insert(std::make_pair((unsigned int)seed,(unsigned int)seed));
    std::stringstream ss1;
    ss1 << "map1_"<<seed;
    m_map1.insert(std::make_pair( ss1.str(),ss1.str() ) );
    m_list.push_back( seed );
    m_set.insert( ss1.str() );
    m_vec2.push_back( Param( seed ) );
    m_map2.insert( std::make_pair( seed, Param( seed ) ) );
    m_vec3.push_back( Param( seed ) );
  }
  
  size_t i=0;
  int j = 1;
  while( j<128 ){
    if( seed & j )m_bitset.set(i);
    j = j << 1;
    i++;
  }
}

bool ArrayPayload::operator ==(const ArrayPayload& rhs) const {
  if( m_i != rhs.m_i ) return false;
  for( int i=0;i<4;i++){
    if(m_ai0[i] != rhs.m_ai0[i] ) return false;
  }
  for( int i=0;i<112;i++){
    if(m_ai1[i] != rhs.m_ai1[i] ) return false;
  }
  for( size_t i=0;i<2; i++ ){
    for( size_t j=0;i<3; i++ ){
      if( m_ai2[j][i] != rhs.m_ai2[j][i] ) return false;
    }
  }
  for( size_t i=0;i<80; i++ ){
    for( size_t j=0;i<2; i++ ){
      if( m_ai3[j][i] != rhs.m_ai3[j][i] ) return false;
    }
  }
  for( size_t i=0;i<4; i++ ){
    if(m_as0[i] != rhs.m_as0[i] ) return false;
  }
  for( size_t i=0;i<112; i++ ){
    if(m_as1[i] != rhs.m_as1[i] ) return false;
  }
  for( int i=0;i<4; i++ ){
    if(m_ap0[i] != rhs.m_ap0[i] ) return false;
  }
  for( int i=0;i<112; i++ ){
    if(m_ap1[i] != rhs.m_ap1[i] ) return false;
  }
  if(m_p0 != rhs.m_p0 ) return false;
  if(m_p1 != rhs.m_p1 ) return false;
  if(m_p2 != rhs.m_p2 ) return false;
  if(m_vec0 != rhs.m_vec0 ) return false;
  if(m_vec1 != rhs.m_vec1 ) return false;
  if(m_vec2 != rhs.m_vec2 ) return false;
  if(m_vec3 != rhs.m_vec3 ) return false;
  if(m_map0 != rhs.m_map0 ) return false;
  if(m_map1 != rhs.m_map1 ) return false;
  if(m_map2 != rhs.m_map2 ) return false;
  if(m_list != rhs.m_list ) return false;
  if(m_set != rhs.m_set ) return false;
  if(m_bitset != rhs.m_bitset ) return false;
  return true;
}
  
bool ArrayPayload::operator !=(const ArrayPayload& rhs) const {
  return !operator==(rhs);
}
