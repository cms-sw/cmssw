#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/SSERot.h"
#include <ostream>
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec3F const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<", "<< v.arr[2] <<')';
}
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec3D const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<", "<< v.arr[2] <<')';
}


std::ostream & operator<<(std::ostream & out, mathSSE::Rot3F const & r){
  return out << r.axis[0] << '\n' << r.axis[1] << '\n' << r.axis[2];
}

std::ostream & operator<<(std::ostream & out, mathSSE::Rot3D const & r){
  return out << r.axis[0] << '\n' << r.axis[1] << '\n' << r.axis[2];
}
