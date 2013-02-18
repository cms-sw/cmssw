
#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVECT
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif

#if defined(USE_EXTVECT)       
#include "DataFormats/Math/interface/ExtVec.h"
#else
#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/SSERot.h"
using namespace mathSSE;
#endif

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

#if defined(USE_EXTVECT)
std::ostream & operator<<(std::ostream & out, ::As3D<Vec4F> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, ::As3D<Vec4D> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}
#else
std::ostream & operator<<(std::ostream & out, As3D<float> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, As3D<double> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}
#endif

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

