#include "CondCore/RegressionTest/interface/PrimitivePayload.h"
#include <sstream>

PrimitivePayload::PrimitivePayload():
  m_i(-1),
  t_bool(false),
  t_uchar(0),
  t_char(-1),
  t_short(-1),
  t_ushort(0),
  t_int(-1),
  t_uint(0),
  t_long(-1),
  t_ulong(0),
  t_llong(-1),
  t_float(0.),
  t_double(0.),
  t_string(""),
  t_enum(A){
}

PrimitivePayload::PrimitivePayload( int seed ):
  m_i(seed + 9999),
  t_bool(false),
  t_uchar(seed),
  t_char(seed),
  t_short(seed),
  t_ushort(seed),
  t_int(seed),
  t_uint(seed),
  t_long(seed),
  t_ulong(seed),
  t_llong(seed),
  t_float(seed),
  t_double(seed),
  t_string(""),
  t_enum(A){
  if( seed % 2== 0 ){
    t_bool = true;
    t_enum = D;
  }
  std::stringstream ss;
  ss << seed;
  t_string = ss.str(); 
}

bool PrimitivePayload::operator ==(const PrimitivePayload& rhs) const {
  if( m_i != rhs.m_i ) return false;
  if( t_bool != rhs.t_bool ) return false;
  if( t_uchar != rhs.t_uchar ) return false;
  if( t_char != rhs.t_char ) return false;
  if( t_short != rhs.t_short ) return false;
  if( t_ushort != rhs.t_ushort ) return false;
  if( t_int != rhs.t_int ) return false;
  if( t_uint != rhs.t_uint ) return false;
  if( t_long != rhs.t_long ) return false;
  if( t_ulong != rhs.t_ulong ) return false;
  if( t_llong != rhs.t_llong ) return false;
  if( t_float != rhs.t_float ) return false;
  if( t_double != rhs.t_double ) return false;
  if( t_string != rhs.t_string ) return false;
  if( t_enum != rhs.t_enum ) return false;
  return true;
}
bool PrimitivePayload::operator !=(const PrimitivePayload& rhs) const {
  return !operator==(rhs);
}
