#ifndef HeterogeneousCore_AlpakaMath_interface_Vector_h
#define HeterogeneousCore_AlpakaMath_interface_Vector_h

// Modified and extended version based on VecArray (author: Felice Pantaleo, CERN)
// Author: Alexei Strelchenko, FNAL
//

#include <utility>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

namespace cms::alpakatools::math {

  template <class T, int maxSize>
  class Vector {
  public:
    // same notations as std::vector/array
    using value_type = T;
    static constexpr int N = maxSize;

    Vector() = default;

    constexpr explicit Vector(const T &value) : m_data{} {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        m_data[i] = value;
      }
    }

    template <typename... U,
              typename = std::enable_if_t<(sizeof...(U) == maxSize) and (std::conjunction_v<std::is_same<T, U>...>)>>
    constexpr Vector(U... args) : m_data{args...} {}

    Vector(const Vector<T, maxSize> &) = default;
    Vector(Vector<T, maxSize> &&) = default;

    Vector<T, maxSize> &operator=(const Vector<T, maxSize> &) = default;
    Vector<T, maxSize> &operator=(Vector<T, maxSize> &&) = default;

    constexpr T const *begin() const { return m_data; }
    constexpr T const *end() const { return m_data + N; }
    constexpr T *begin() { return m_data; }
    constexpr T *end() { return m_data + N; }
    constexpr int size() const { return N; }
    constexpr T &operator[](int i) { return m_data[i]; }
    constexpr const T &operator[](int i) const { return m_data[i]; }
    constexpr T const *data() const { return m_data; }

    constexpr void zero() {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++)
        m_data[i] = static_cast<T>(0);
    }

    constexpr T norm2() const {
      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        res += m_data[i] * m_data[i];
      }
      return res;
    }

    template <unsigned int n>
    constexpr T partial_norm2() const {
      constexpr int reduced_size = n > maxSize ? maxSize : n;

      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < reduced_size; i++) {
        res += m_data[i] * m_data[i];
      }

      return res;
    }

    constexpr Vector<T, maxSize> &operator*=(const T &scale) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; ++i) {
        m_data[i] *= scale;
      }
      return *this;
    }

    constexpr Vector<T, maxSize> &operator+=(const Vector<T, maxSize> &rhs) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; ++i) {
        m_data[i] += rhs[i];
      }
      return *this;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC T norm(const TAcc &acc) const {
      const T nrm2 = norm2();
      return alpaka::math::sqrt(acc, nrm2);
    }

    template <typename TAcc, unsigned int n>
    ALPAKA_FN_ACC T partial_norm(const TAcc &acc) const {
      const T partial_nrm2 = partial_norm2<n>();
      return alpaka::math::sqrt(acc, partial_nrm2);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC void normalize(const TAcc &acc) {
      const T nrm = norm(acc);

      if (nrm == static_cast<T>(0))
        return;

      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        m_data[i] /= nrm;
      }
    }

  private:
    T m_data[maxSize];
  };

  template <typename T, int N>
  inline constexpr Vector<T, N> zero() {
    Vector<T, N> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = static_cast<T>(0);
    }

    return res;
  }

  template <typename U, typename T, int N>
    requires std::convertible_to<U, typename Vector<T, N>::value_type>
  inline constexpr Vector<T, N> scale(const U a, const Vector<T, N> &x) {
    Vector<T, N> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = a * x[i];
    }

    return res;
  }

  template <typename T, int N>
  inline constexpr Vector<T, N> add(const Vector<T, N> &x, const Vector<T, N> &y) {
    Vector<T, N> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = x[i] + y[i];
    }

    return res;
  }

  template <typename T, int N>
  inline constexpr Vector<T, N> sub(const Vector<T, N> &x, const Vector<T, N> &y) {
    Vector<T, N> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = x[i] - y[i];
    }

    return res;
  }

  template <typename U, typename T, int N>
    requires std::convertible_to<U, typename Vector<T, N>::value_type>
  inline constexpr Vector<T, N> axpy(const U a, const Vector<T, N> &x, const Vector<T, N> &y) {
    Vector<T, N> res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res[i] = a * x[i] + y[i];
    }

    return res;
  }

  template <typename T, int N>
  inline constexpr T dot(const Vector<T, N> &x, const Vector<T, N> &y) {
    T res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      res += x[i] * y[i];
    }

    return res;
  }

  template <typename T, int N>
  inline constexpr T diff2(const Vector<T, N> &x, const Vector<T, N> &y) {
    T res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < N; i++) {
      const T tmp = x[i] - y[i];
      res += tmp * tmp;
    }
    return res;
  }

}  // namespace cms::alpakatools::math

#endif  // HeterogeneousCore_AlpakaMath_interface_Vector_h
