#include <string>

class PrimitivePayload {
  public:
  PrimitivePayload();
  PrimitivePayload( int seed );
  bool operator ==(const PrimitivePayload& rhs) const;
  bool operator !=(const PrimitivePayload& rhs) const;
  private:
  int m_i;
  bool t_bool;
  unsigned char t_uchar;
  char t_char;
  short t_short;
  unsigned short t_ushort;
  int t_int;
  unsigned int t_uint;
  long t_long;
  unsigned long t_ulong;
  long long t_llong;
  float t_float;
  double t_double;
  std::string t_string;
  enum T_Enum { A =3, B, C= 101, D, E, F};
  T_Enum t_enum;
};
