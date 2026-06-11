#ifndef DataFormats_EgammaReco_interface_alpaka_Phys3DVector_h
#define DataFormats_EgammaReco_interface_alpaka_Phys3DVector_h

#include <utility>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

namespace cms::alpakatools::math {

  template <class T>
  class Phys3DVector {
  public:
    using value_type = T;

    constexpr Phys3DVector() : m_data{} {};

    Phys3DVector(const Phys3DVector<T>&) = default;

    constexpr Phys3DVector(const T& value) : m_data{} {
      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; i++) {
        m_data[i] = value;
      }
    }

    constexpr Phys3DVector(const T x, const T y, const T z) : m_data{} {
      m_data[0] = x;
      m_data[1] = y;
      m_data[2] = z;
    }

    Phys3DVector<T>& operator=(const Phys3DVector<T>&) = default;

    inline constexpr T& operator[](int i) { return m_data[i]; }
    inline constexpr const T& operator[](int i) const { return m_data[i]; }

    inline constexpr int size() const { return 3; }

    // Extra:
    inline constexpr void zero() {
      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; i++) {
        m_data[i] = static_cast<T>(0);
      }
    }

    inline constexpr T norm2() const {
      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; i++) {
        res += m_data[i] * m_data[i];
      }
      return res;
    }

    inline constexpr T partial_norm2() const {
      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < 2; i++) {
        res += m_data[i] * m_data[i];
      }

      return res;
    }

    inline constexpr Phys3DVector<T>& operator*=(const T& scale) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; ++i) {
        m_data[i] *= scale;
      }
      return *this;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC T norm(const TAcc& acc) const {
      const T nrm2 = norm2();
      return alpaka::math::sqrt(acc, nrm2);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC T partial_norm(const TAcc& acc) const {
      const T partial_nrm2 = partial_norm2();
      return alpaka::math::sqrt(acc, partial_nrm2);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC void normalize(const TAcc& acc) {
      const T nrm = norm(acc);

      if (nrm == 0.)
        return;

      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; i++) {
        m_data[i] /= nrm;
      }
    }

  private:
    T m_data[3];
  };

  template <typename T>
  inline constexpr Phys3DVector<T> ax(const T a, const Phys3DVector<T>& x) {
    Phys3DVector<T> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      res[i] = a * x[i];
    }

    return res;
  }

  template <typename T>
  inline constexpr Phys3DVector<T> xmy(const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    Phys3DVector<T> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      res[i] = x[i] - y[i];
    }

    return res;
  }

  template <typename T>
  inline constexpr Phys3DVector<T> axpy(const T a, const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    Phys3DVector<T> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      res[i] = a * x[i] + y[i];
    }

    return res;
  }

  template <typename T>
  inline constexpr T dot(const Phys3DVector<T>& x, const Phys3DVector<T>& y) {
    T res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < 3; i++) {
      res += x[i] * y[i];
    }

    return res;
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

}  // namespace cms::alpakatools::math

#endif
