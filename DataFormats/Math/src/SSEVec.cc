#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/SSERot.h"
#include <ostream>
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec4F const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<", "<< v.arr[2] <<", "<< v.arr[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec4D const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<", "<< v.arr[2] <<", "<< v.arr[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec2D const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<')';
}

std::ostream & operator<<(std::ostream & out, mathSSE::As3D<float> const & v) {
  return out << '(' << v.v.arr[0] <<", " << v.v.arr[1] <<", "<< v.v.arr[2] <<')';
}

std::ostream & operator<<(std::ostream & out, mathSSE::As3D<double> const & v) {
  return out << '(' << v.v.arr[0] <<", " << v.v.arr[1] <<", "<< v.v.arr[2] <<')';
}

std::ostream & operator<<(std::ostream & out, mathSSE::Rot3F const & r){
  return out << as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

std::ostream & operator<<(std::ostream & out, mathSSE::Rot3D const & r){
  return out <<  as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

