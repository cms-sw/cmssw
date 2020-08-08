#ifndef RecoPixelVertexingPixelTripletsCircleEq_H
#define RecoPixelVertexingPixelTripletsCircleEq_H
/**
| 1) circle is parameterized as:                                              |
|    C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             |
|    Xp,Yp is a point on the track;                                           |
|    C = 1/r0 is the curvature  ( sign of C is charge of particle );          |
|    alpha & beta are the direction cosines of the radial vector at Xp,Yp     |
|    i.e.  alpha = C*(X0-Xp),                                                 |
|          beta  = C*(Y0-Yp),                                                 |
|    where center of circle is at X0,Y0.                                      |
|                                                                             |
|    Slope dy/dx of tangent at Xp,Yp is -alpha/beta.                          |
| 2) the z dimension of the helix is parameterized by gamma = dZ/dSperp       |
|    this is also the tangent of the pitch angle of the helix.                |
|    with this parameterization, (alpha,beta,gamma) rotate like a vector.     |
| 3) For tracks going inward at (Xp,Yp), C, alpha, beta, and gamma change sign|
|
*/

#include <cmath>

template <typename T>
class CircleEq {
public:
  CircleEq() {}

  constexpr CircleEq(T x1, T y1, T x2, T y2, T x3, T y3) { compute(x1, y1, x2, y2, x3, y3); }

  constexpr void compute(T x1, T y1, T x2, T y2, T x3, T y3);

  // dca to origin divided by curvature
  constexpr T dca0() const {
    auto x = m_c * m_xp + m_alpha;
    auto y = m_c * m_yp + m_beta;
    return std::sqrt(x * x + y * y) - T(1);
  }

  // dca to given point (divided by curvature)
  constexpr T dca(T x, T y) const {
    x = m_c * (m_xp - x) + m_alpha;
    y = m_c * (m_yp - y) + m_beta;
    return std::sqrt(x * x + y * y) - T(1);
  }

  // curvature
  constexpr auto curvature() const { return m_c; }

  // alpha and beta
  constexpr std::pair<T, T> cosdir() const { return std::make_pair(m_alpha, m_beta); }

  // alpha and beta af given point
  constexpr std::pair<T, T> cosdir(T x, T y) const {
    return std::make_pair(m_alpha - m_c * (x - m_xp), m_beta - m_c * (y - m_yp));
  }

  // center
  constexpr std::pair<T, T> center() const { return std::make_pair(m_xp + m_alpha / m_c, m_yp + m_beta / m_c); }

  constexpr auto radius() const { return T(1) / m_c; }

  T m_xp = 0;
  T m_yp = 0;
  T m_c = 0;
  T m_alpha = 0;
  T m_beta = 0;
};

template <typename T>
constexpr void CircleEq<T>::compute(T x1, T y1, T x2, T y2, T x3, T y3) {
  bool noflip = std::abs(x3 - x1) < std::abs(y3 - y1);

  auto x1p = noflip ? x1 - x2 : y1 - y2;
  auto y1p = noflip ? y1 - y2 : x1 - x2;
  auto d12 = x1p * x1p + y1p * y1p;
  auto x3p = noflip ? x3 - x2 : y3 - y2;
  auto y3p = noflip ? y3 - y2 : x3 - x2;
  auto d32 = x3p * x3p + y3p * y3p;

  auto num = x1p * y3p - y1p * x3p;  // num also gives correct sign for CT
  auto det = d12 * y3p - d32 * y1p;

  /*
  auto ct  = num/det;
  auto sn  = det>0 ? T(1.) : T(-1.);
  auto st2 = (d12*x3p-d32*x1p)/det;
  auto seq = T(1.) +st2*st2;
  auto al2 = sn/std::sqrt(seq);
  auto be2 = -st2*al2;
  ct *= T(2.)*al2;
  */

  auto st2 = (d12 * x3p - d32 * x1p);
  auto seq = det * det + st2 * st2;
  auto al2 = T(1.) / std::sqrt(seq);
  auto be2 = -st2 * al2;
  auto ct = T(2.) * num * al2;
  al2 *= det;

  m_xp = x2;
  m_yp = y2;
  m_c = noflip ? ct : -ct;
  m_alpha = noflip ? al2 : -be2;
  m_beta = noflip ? be2 : -al2;
}

#endif
