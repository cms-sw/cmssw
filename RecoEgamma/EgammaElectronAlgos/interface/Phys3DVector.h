#ifndef RecoEgamma_EgammaElectronAlgos_interface_Phys3DVector_h
#define RecoEgamma_EgammaElectronAlgos_interface_Phys3DVector_h

#include <utility>

#include <alpaka/alpaka.hpp>

#include <limits>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

namespace egamma::math {

  // Represents a vector in three-dimensional space, expressed using either
  // Cartesian coordinates (x, y, z) or cylindrical coordinates (rho, z).

  template <class T>
  class Phys3DVector {
  public:
    using value_type = T;

    constexpr Phys3DVector() : m_data{} {};

    Phys3DVector(const Phys3DVector<T>&) = default;

    constexpr Phys3DVector(const T value) : m_data{value, value, value} {}

    constexpr Phys3DVector(const T x, const T y, const T z) : m_data{x, y, z} {}

    Phys3DVector<T>& operator=(const Phys3DVector<T>&) = default;

    inline constexpr T& operator[](int i) { return m_data[i]; }
    inline constexpr const T& operator[](int i) const { return m_data[i]; }

    inline constexpr int size() const { return 3; }

    // Extra:
    inline constexpr void zero() {
      m_data[0] = static_cast<T>(0);
      m_data[1] = static_cast<T>(0);
      m_data[2] = static_cast<T>(0);
    }

    inline constexpr T r2() const { return (m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]); }

    inline constexpr T rho2() const { return (m_data[0] * m_data[0] + m_data[1] * m_data[1]); }

    inline constexpr Phys3DVector<T>& operator*=(const T& scale) {
      m_data[0] *= scale;
      m_data[1] *= scale;
      m_data[2] *= scale;
      return *this;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC T r(const TAcc& acc) const {
      return alpaka::math::sqrt(acc, r2());
    }

    template <typename TAcc>
    ALPAKA_FN_ACC T rho(const TAcc& acc) const {
      return alpaka::math::sqrt(acc, rho2());
    }

    template <typename TAcc>
    ALPAKA_FN_ACC void normalize(const TAcc& acc) {
      const T mag = r(acc);

      if (mag < std::numeric_limits<T>::epsilon())
        return;

      m_data[0] /= mag;
      m_data[1] /= mag;
      m_data[2] /= mag;
    }

  private:
    T m_data[3];
  };

  template <typename T>
  inline constexpr Phys3DVector<T> operator*(const T a, const Phys3DVector<T>& x) {
    return Phys3DVector<T>{a * x[0], a * x[1], a * x[2]};
  }

  template <typename T>
  inline constexpr Phys3DVector<T> operator-(const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    return Phys3DVector<T>{x[0] - y[0], x[1] - y[1], x[2] - y[2]};
  }

  template <typename TAcc, typename T>
  inline constexpr Phys3DVector<T> axpy(TAcc const& acc, const T a, const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    Phys3DVector<T> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      res[i] = alpaka::math::fma(acc, a, x[i], y[i]);
    }

    return res;
  }

  template <typename T>
  inline constexpr T operator*(const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
  }

  template <typename T>
  inline constexpr T diff_norm2(const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    T res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      const T tmp = x[i] - y[i];
      res += tmp * tmp;
    }
    return res;
  }

  template <typename T>
  inline constexpr T diff_dot(const Phys3DVector<T>& x, const Phys3DVector<T>& y, const Phys3DVector<T>& z) {
    T res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      const T tmp = x[i] * (y[i] - z[i]);
      res += tmp * tmp;
    }
    return res;
  }

}  // namespace egamma::math

#endif  // RecoEgamma_EgammaElectronAlgos_interface_Phys3DVector_h
