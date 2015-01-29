#if ((defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)) && !defined(__INTEL_COMPILER)
#include "DataFormats/Math/interface/ExtVec.h"

#include <ostream>
std::ostream & operator<<(std::ostream & out,  Vec4F const & v) {
  return out << '(' << v[0] <<", " << v[1] <<", "<< v[2] <<", "<< v[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  Vec4D const & v) {
  return out << '(' << v[0] <<", " << v[1] <<", "<< v[2] <<", "<< v[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  Vec2F const & v) {
  return out << '(' << v[0] <<", " << v[1] <<')';
}
std::ostream & operator<<(std::ostream & out,  Vec2D const & v) {
  return out << '(' << v[0] <<", " << v[1] <<')';
}

std::ostream & operator<<(std::ostream & out, ::As3D<Vec4F> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, ::As3D<Vec4D> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, Rot3F const & r){
  return out << as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

std::ostream & operator<<(std::ostream & out, Rot3D const & r){
  return out <<  as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

std::ostream & operator<<(std::ostream & out, Rot2F const & r){
  return out << r.axis[0] << '\n' << r.axis[1];
}

std::ostream & operator<<(std::ostream & out, Rot2D const & r){
  return out << r.axis[0] << '\n' << r.axis[1];
}
#endif
