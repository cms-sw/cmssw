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

  template <typename T>
  constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> zero() {
    return static_cast<T>(0);
  }

  template <typename T, typename U>
  constexpr std::enable_if_t<std::is_arithmetic_v<T> and std::is_arithmetic_v<U>, T> set(U x) {
    return static_cast<T>(x);
  }

  template <class T, int maxSize>
  class Vector {
  public:
    // same notations as std::vector/array
    using value_type = T;
    static constexpr int N = maxSize;

    constexpr int push_back_unsafe(const T &element) {
      if (m_size < maxSize) {
        const auto prev_size = m_size;
        m_data[m_size] = element;
        m_size += 1;
        return prev_size;
      } 
      return -1;
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&...args) {
      if (m_size < maxSize) {
        const auto prev_size = m_size;
        m_data[m_size] = T(std::forward<Ts>(args)...);
        m_size += 1;
        return prev_size;
      } 
      return -1;
    }

    constexpr T const &back() const {
      assert(m_size > 0 && "Vector::back() on empty vector");
      return m_data[m_size - 1];
    }

    constexpr T &back() {
      assert(m_size > 0 && "Vector::back() on empty vector");
      return m_data[m_size - 1];
    }

    // thread-safe version of the vector, when used in a kernel
    template <typename TAcc>
    ALPAKA_FN_ACC int push_back(const TAcc &acc, const T &element) {
      const auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    template <typename TAcc, class... Ts>
    ALPAKA_FN_ACC int emplace_back(const TAcc &acc, Ts &&...args) {
      const auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        m_data[previousSize] = T(std::forward<Ts>(args)...);
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    constexpr T pop_back() {
      if (m_size > 0) {
        auto previousSize = m_size--;
        return m_data[previousSize - 1];
      } else
        return T();
    }

    constexpr Vector() : m_data{}, m_size(maxSize) { };

    constexpr explicit Vector(const T &value) : m_data{}, m_size(maxSize) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        m_data[i] = value;
      }
    }

    template<typename... U, typename = std::enable_if_t<(sizeof...(U) == maxSize) and (std::conjunction_v<std::is_same<T, U>...>)>>
    constexpr Vector(U... args) : m_data{ args... }, m_size(sizeof...(U)) {}

    Vector(const Vector<T, maxSize> &) = default;
    Vector(Vector<T, maxSize> &&) = default;

    Vector<T, maxSize> &operator=(const Vector<T, maxSize> &) = default;
    Vector<T, maxSize> &operator=(Vector<T, maxSize> &&) = default;

    constexpr T const *begin() const { return m_data; }
    constexpr T const *end() const { return m_data + m_size; }
    constexpr T *begin() { return m_data; }
    constexpr T *end() { return m_data + m_size; }
    constexpr int size() const { return m_size; }
    constexpr T &operator[](int i) { return m_data[i]; }
    constexpr const T &operator[](int i) const { return m_data[i]; }
    constexpr void reset() { m_size = 0; }
    static constexpr int capacity() { return maxSize; }
    constexpr T const *data() const { return m_data; }
    constexpr void resize(int size) { m_size = size; }
    constexpr bool empty() const { return 0 == m_size; }
    constexpr bool full() const { return maxSize == m_size; }

    constexpr void zero() {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++)
        m_data[i] = zero<T>();
    }

    constexpr T norm2() const {
      T res{0};	    
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        res += m_data[i]*m_data[i];
      }
      return res;
    }


    template<unsigned int n>
    constexpr T partial_norm2() const {
      constexpr int reduced_size = n > maxSize ? maxSize : n;

      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < reduced_size; i++) {
        res += m_data[i]*m_data[i];
      }

      return res;
    }
    
    constexpr Vector<T, maxSize>& operator*=(const T& scale) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; ++i) {
        m_data[i] *= scale;
      }
      return *this;
    }

    constexpr Vector<T, maxSize>& operator+=(const Vector<T, maxSize>& rhs) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; ++i) {
        m_data[i] += rhs[i];
      }
      return *this;
    }    

    template <typename TAcc >
    ALPAKA_FN_ACC T norm( const TAcc &acc ) const {
      const T nrm2 = norm2();
      return alpaka::math::sqrt(acc, nrm2);
    }    

    template <typename TAcc, unsigned int n >
    ALPAKA_FN_ACC T partial_norm( const TAcc &acc) const {

      const T partial_nrm2 = partial_norm2<n>();
      return alpaka::math::sqrt(acc, partial_nrm2);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC void normalize( const TAcc &acc) const {

      const T nrm = norm(acc);

      if (nrm == static_cast<T>(0)) return;

      CMS_UNROLL_LOOP
      for (int i = 0; i < maxSize; i++) {
        m_data[i] /= nrm;
      }
    }

  private:
    T m_data[maxSize];

    int m_size;
  };

  template <typename T>
  struct is_Vector : std::false_type {};

  template <typename T, int N>
  struct is_Vector<cms::alpakatools::math::Vector<T, N>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_Vector_v = is_Vector<T>::value;

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN zero(){
    VectorN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < VectorN::N; i++) {
      res[i] = zero<typename VectorN::value_type>();
    }

    return res;
  } 

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN ax(const typename VectorN::value_type a, const VectorN& x){
    VectorN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = a*x[i];
    }

    return res;
  }  

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN xpy( const VectorN& x, const VectorN& y ){
    VectorN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] + y[i];
    }

    return res;
  }

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN xmy( const VectorN& x, const VectorN& y ){
    VectorN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] - y[i];
    }

    return res;
  }

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN axpy( const typename VectorN::value_type a, const VectorN& x, const VectorN& y ){
    VectorN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = a * x[i] + y[i];
    }

    return res;
  }

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN::value_type dot( const VectorN& x, const VectorN& y ){
    typename VectorN::value_type res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res += x[i] * y[i];
    }

    return res;
  }  

  template <typename VectorN, std::enable_if_t<is_Vector_v<VectorN>, int> = 0>
  inline constexpr VectorN::value_type diff2( const VectorN& x, const VectorN& y ){
    typename VectorN::value_type res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      const typename VectorN::value_type tmp = x[i] - y[i];
      res += tmp*tmp;
    }
    return res;
  }

}  // namespace cms::alpakatools::math

#endif  // HeterogeneousCore_AlpakaMath_interface_Vector_h
