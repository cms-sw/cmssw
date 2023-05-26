#include "DataFormats/Math/interface/ExtVec.h"

#include <cmath>
#include <vector>

#include <iostream>

#ifdef __AVX2__
#define CMS_USE_AVX2
#endif /* __AVX2__ */

void addScaleddiff(Vec3F& res, float s, Vec3F const& a, Vec3F const& b) { res = res + s * (a - b); }

float dotV(Vec3F const& a, Vec3F const& b) { return dot(a, b); }

float dotSimple(Vec3F const& a, Vec3F const& b) {
  Vec3F res = a * b;
  return res[0] + res[1] + res[2];
}

double dotSimple(Vec3D const& a, Vec3D const& b) {
  Vec3D res = a * b;
  return res[0] + res[1] + res[2];
}

float norm(Vec3F const& a) { return std::sqrt(dot(a, a)); }

Vec3F toLocal(Vec3F const& a, Rot3<float> const& r) { return r.rotate(a); }

Vec3F toGlobal(Vec3F const& a, Rot3<float> const& r) { return r.rotateBack(a); }

// fake basicVector to check constructs...
template <typename T>
struct BaVec {
  typedef BaVec<T> self;

  BaVec() : theX(0), theY(0), theZ(0), theW(0) {}

  BaVec(float f1, float f2, float f3) : theX(f1), theY(f2), theZ(f3), theW(0) {}

  self& operator+=(self const& rh) { return *this; }

  T theX;
  T theY;
  T theZ;
  T theW;
} __attribute__((aligned(16)));

typedef BaVec<float> BaVecF;

static_assert(sizeof(BaVecF) == sizeof(Vec3F));

struct makeVec3F {
  makeVec3F(BaVecF& bv) : v(reinterpret_cast<Vec3F&>(bv)) {}
  Vec3F& v;
};
struct makeVec3FC {
  makeVec3FC(BaVecF const& bv) : v(reinterpret_cast<Vec3F const&>(bv)) {}
  Vec3F const& v;
};

template <>
inline BaVecF& BaVecF::operator+=(BaVecF const& rh) {
  makeVec3FC v(rh);
  makeVec3F s(*this);
  s.v = s.v + v.v;
  return *this;
}

void sum(BaVecF& lh, BaVecF const& rh) { lh += rh; }

void testBa() {
  std::cout << " test BA of size " << sizeof(BaVecF) << std::endl;
  BaVecF vx(2.0, 4.0, 5.0);
  BaVecF vy(-3.0, 2.0, -5.0);
  vx += vy;
  std::cout << vx.theX << ", " << vx.theY << ", " << vx.theZ << std::endl;
}

template <typename T>
void go2d() {
  typedef Vec2<T> Vec2d;
  typedef Vec4<T> Vec3d;
  static_assert(sizeof(Vec2d) == 2 * sizeof(T));
  static_assert(sizeof(Vec3d) == 4 * sizeof(T));

  std::cout << "\n2d" << std::endl;
  std::cout << sizeof(Vec2d) << ' ' << alignof(Vec2d) << std::endl;

  std::vector<Vec2d> vec1;
  vec1.reserve(50);
  std::vector<T> vect(23);
  std::vector<Vec2d> vec2(53);
  std::vector<Vec2d> vec3;
  vec3.reserve(50234);

  constexpr Vec2d k{-2.0, 3.14};
  std::cout << k << std::endl;
  std::cout << k + k << std::endl;
  std::cout << k * k << std::endl;
  Vec3d x{2.0, 4.0, 5.0};
  Vec3d y{-3.0, 2.0, -5.0};
  std::cout << x << std::endl;
  std::cout << y << std::endl;

  Vec2d x2 = xy(x);
  Vec2d y2 = xy(y);
  std::cout << x2 << std::endl;
  Vec2d xx2 = xy(x);
  std::cout << xx2 << std::endl;
  std::cout << y2 << std::endl;

  std::cout << Vec2d(T(3.) * x2) << std::endl;
  std::cout << Vec2d(y2 * T(0.1)) << std::endl;
  std::cout << Vec2d(T(0.5) * (x2 + y2)) << std::endl;
  std::cout << apply(x2, [](T x) { return std::sqrt(x); }) << std::endl;

  std::cout << dot(x2, y2) << " = 2?" << std::endl;
  std::cout << dot2(x2, y) << " = 2?" << std::endl;
  std::cout << dot2(x, y) << " = 2?" << std::endl;

  T z = cross2(x2, y2);
  std::cout << z << " = 16?" << std::endl;
  z = cross2(x, y);
  std::cout << z << " = 16?" << std::endl;

  std::cout << std::sqrt(z) << " = 4?" << std::endl;
}

template <typename T>
void go(bool dovec = true) {
  typedef Vec4<T> Vec;
  typedef Vec2<T> Vec2D;

  std::cout << std::endl;
  std::cout << sizeof(Vec) << ' ' << alignof(Vec) << std::endl;

  if (dovec) {
    std::vector<Vec> vec1;
    vec1.reserve(50);
    std::vector<T> vect(23);
    std::vector<Vec> vec2(53);
    std::vector<Vec> vec3;
    vec3.reserve(50234);
  }

  constexpr Vec zero{0, 0, 0, 0};
  constexpr Vec x{2.0f, 4.0f, 5.0f};
  constexpr Vec y{-3.0f, 2.0f, -5.0f};
  Vec x0 = x[0] + zero;
  //constexpr Vec xx = T(3.3) + zero;  // clang 3.8 does not like it
  const Vec xx = T(3.3) + zero;
  std::cout << x << std::endl;
  std::cout << (Vec4<float>){float(x[0]), float(x[1]), float(x[2]), float(x[3])} << std::endl;
  std::cout << (Vec4<double>){x[0], x[1], x[2], x[3]} << std::endl;
  std::cout << -x << std::endl;
  std::cout << Vec{x[2]} << std::endl;
  //std::cout <<  Vec(x[2]) << std::endl;
  std::cout << x0 << std::endl;
  std::cout << xx << std::endl;
  std::cout << Vec{} + x[2] << std::endl;
  std::cout << y << std::endl;
  std::cout << T(3.) * x << std::endl;
  std::cout << y * T(0.1) << std::endl;
  std::cout << (T(1) - y * T(0.1)) << std::endl;
  std::cout << apply(x, [](T x) { return std::sqrt(x); }) << std::endl;

  std::cout << dot(x, y) << std::endl;
  std::cout << dotSimple(x, y) << std::endl;

  //  std::cout << "equal" << (x==x ? " " : " not ") << "ok" << std::endl;
  // std::cout << "not equal" << (x==y ? " not " : " ") << "ok" << std::endl;

  Vec z = cross3(x, y);
  std::cout << z << std::endl;
  std::cout << cross2(x, y) << std::endl;

  std::cout << "\nrotations" << std::endl;

  // constexpr T a = 0.01;
  constexpr T ca = 0.9999500004166653;    // std::cos(a);  clang does not support constepxr math functions....
  constexpr T sa = 0.009999833334166664;  // std::sin(a);

  constexpr Rot3<T> r1(ca, sa, 0, -sa, ca, 0, 0, 0, 1);

  constexpr Rot2<T> r21(ca, sa, -sa, ca);

  constexpr Rot3<T> r2((Vec){0, 1, 0, 0}, (Vec){0, 0, 1, 0}, (Vec){1, 0, 0, 0});
  constexpr Rot2<T> r22((Vec2D){0, 1}, (Vec2D){1, 0});

  {
    std::cout << "\n3D rot" << std::endl;
    /* constexpr */ Vec xr = r1.rotate(x);
    std::cout << x << std::endl;
    std::cout << xr << std::endl;
    std::cout << r1.rotateBack(xr) << std::endl;

    Rot3<T> rt = r1.transpose();
    Vec xt = rt.rotate(xr);
    std::cout << x << std::endl;
    std::cout << xt << std::endl;
    std::cout << rt.rotateBack(xt) << std::endl;

    std::cout << r1 << std::endl;
    std::cout << rt << std::endl;
    std::cout << r1 * rt << std::endl;
    std::cout << r2 << std::endl;
    std::cout << r1 * r2 << std::endl;
    std::cout << r2 * r1 << std::endl;
    std::cout << r1 * r2.transpose() << std::endl;
    std::cout << r1.transpose() * r2 << std::endl;
  }

  {
    std::cout << "\n2D rot" << std::endl;
    Vec2D xr = r21.rotate(xy(x));
    std::cout << xy(x) << std::endl;
    std::cout << xr << std::endl;
    std::cout << r21.rotateBack(xr) << std::endl;

    Rot2<T> rt = r21.transpose();
    Vec2D xt = rt.rotate(xr);
    std::cout << xy(x) << std::endl;
    std::cout << xt << std::endl;
    std::cout << rt.rotateBack(xt) << std::endl;

    std::cout << r21 << std::endl;
    std::cout << rt << std::endl;
    std::cout << r21 * rt << std::endl;
    std::cout << r22 << std::endl;
    std::cout << r21 * r22 << std::endl;
    std::cout << r22 * r21 << std::endl;
    std::cout << r21 * r22.transpose() << std::endl;
    std::cout << r21.transpose() * r22 << std::endl;
  }
}

int main() {
#ifdef CMS_USE_AVX2
  std::cout << "using AVX" << std::endl;
#endif
  testBa();
  go<float>();
#ifdef CMS_USE_AVX2
  go<double>(false);
#else
  go<double>();
#endif
  go2d<float>();
  go2d<double>();

  return 0;
}
