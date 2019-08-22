#ifndef DataFormat_Math_SSERot_H
#define DataFormat_Math_SSERot_H

#include "DataFormats/Math/interface/SSEVec.h"

namespace mathSSE {

  template <typename T>
  struct OldRot {
    T R11, R12, R13;
    T R21, R22, R23;
    T R31, R32, R33;
  } __attribute__((aligned(16)));

  template <typename T>
  struct Rot3 {
    Vec4<T> axis[3];

    Rot3() {
      axis[0].arr[0] = 1;
      axis[1].arr[1] = 1;
      axis[2].arr[2] = 1;
    }

    Rot3(Vec4<T> ix, Vec4<T> iy, Vec4<T> iz) {
      axis[0] = ix;
      axis[1] = iy;
      axis[2] = iz;
    }

    Rot3(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) {
      axis[0].set(xx, xy, xz);
      axis[1].set(yx, yy, yz);
      axis[2].set(zx, zy, zz);
    }

    Rot3 transpose() const {
      return Rot3(axis[0].arr[0],
                  axis[1].arr[0],
                  axis[2].arr[0],
                  axis[0].arr[1],
                  axis[1].arr[1],
                  axis[2].arr[1],
                  axis[0].arr[2],
                  axis[1].arr[2],
                  axis[2].arr[2]);
    }

    Vec4<T> x() { return axis[0]; }
    Vec4<T> y() { return axis[1]; }
    Vec4<T> z() { return axis[2]; }

    // toLocal...
    Vec4<T> rotate(Vec4<T> v) const { return transpose().rotateBack(v); }

    // toGlobal...
    Vec4<T> rotateBack(Vec4<T> v) const {
      return v.template get1<0>() * axis[0] + v.template get1<1>() * axis[1] + v.template get1<2>() * axis[2];
    }

    Rot3 rotate(Rot3 const& r) const {
      Rot3 tr = transpose();
      return Rot3(tr.rotateBack(r.axis[0]), tr.rotateBack(r.axis[1]), tr.rotateBack(r.axis[2]));
    }

    Rot3 rotateBack(Rot3 const& r) const {
      return Rot3(rotateBack(r.axis[0]), rotateBack(r.axis[1]), rotateBack(r.axis[2]));
    }
  };

  typedef Rot3<float> Rot3F;

  typedef Rot3<double> Rot3D;

#ifdef __SSE4_1__
  template <>
  inline Vec4<float> Rot3<float>::rotate(Vec4<float> v) const {
    return _mm_or_ps(_mm_or_ps(_mm_dp_ps(axis[0].vec, v.vec, 0x71), _mm_dp_ps(axis[1].vec, v.vec, 0x72)),
                     _mm_dp_ps(axis[2].vec, v.vec, 0x74));
  }
  template <>
  inline Rot3<float> Rot3<float>::rotate(Rot3<float> const& r) const {
    return Rot3<float>(rotate(r.axis[0]), rotate(r.axis[1]), rotate(r.axis[2]));
  }

#endif

}  // namespace mathSSE

template <typename T>
inline mathSSE::Rot3<T> operator*(mathSSE::Rot3<T> const& rh, mathSSE::Rot3<T> const& lh) {
  //   return Rot3(lh.rotateBack(rh.axis[0]),lh.rotateBack(rh.axis[1]),lh.rotateBack(rh.axis[2]));
  return lh.rotateBack(rh);
}

namespace mathSSE {

  template <typename T>
  struct Rot2 {
    Vec2<T> axis[2];

    Rot2() {
      axis[0].arr[0] = 1;
      axis[1].arr[1] = 1;
    }

    Rot2(Vec2<T> ix, Vec2<T> iy) {
      axis[0] = ix;
      axis[1] = iy;
    }

    Rot2(T xx, T xy, T yx, T yy) {
      axis[0].set(xx, xy);
      axis[1].set(yx, yy);
    }

    Rot2 transpose() const { return Rot2(axis[0].arr[0], axis[1].arr[0], axis[0].arr[1], axis[1].arr[1]); }

    Vec2<T> x() { return axis[0]; }
    Vec2<T> y() { return axis[1]; }

    // toLocal...
    Vec2<T> rotate(Vec2<T> v) const { return transpose().rotateBack(v); }

    // toGlobal...
    Vec2<T> rotateBack(Vec2<T> v) const { return v.template get1<0>() * axis[0] + v.template get1<1>() * axis[1]; }

    Rot2 rotate(Rot2 const& r) const {
      Rot2 tr = transpose();
      return Rot2(tr.rotateBack(r.axis[0]), tr.rotateBack(r.axis[1]));
    }

    Rot2 rotateBack(Rot2 const& r) const { return Rot2(rotateBack(r.axis[0]), rotateBack(r.axis[1])); }
  };

  typedef Rot2<float> Rot2F;

  typedef Rot2<double> Rot2D;

}  // namespace mathSSE

template <typename T>
inline mathSSE::Rot2<T> operator*(mathSSE::Rot2<T> const& rh, mathSSE::Rot2<T> const& lh) {
  return lh.rotateBack(rh);
}

#include <iosfwd>
std::ostream& operator<<(std::ostream& out, mathSSE::Rot3F const& v);
std::ostream& operator<<(std::ostream& out, mathSSE::Rot3D const& v);
std::ostream& operator<<(std::ostream& out, mathSSE::Rot2F const& v);
std::ostream& operator<<(std::ostream& out, mathSSE::Rot2D const& v);

#endif  //  DataFormat_Math_SSERot_H
