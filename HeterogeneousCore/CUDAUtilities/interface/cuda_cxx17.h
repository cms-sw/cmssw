#ifndef HeterogeneousCore_CUDAUtilities_cuda_cxx17_h
#define HeterogeneousCore_CUDAUtilities_cuda_cxx17_h

#include <initializer_list>

// CUDA does not support C++17 yet, so we define here some of the missing library functions
#if __cplusplus <= 201402L

namespace std {

  // from https://en.cppreference.com/w/cpp/iterator/size
  template <class C>
  constexpr auto size(const C& c) -> decltype(c.size())
  {
    return c.size();
  }

  template <class T, std::size_t N>
  constexpr std::size_t size(const T (&array)[N]) noexcept
  {
    return N;
  }

  // from https://en.cppreference.com/w/cpp/iterator/empty
  template <class C>
  constexpr auto empty(const C& c) -> decltype(c.empty())
  {
    return c.empty();
  }

  template <class T, std::size_t N>
  constexpr bool empty(const T (&array)[N]) noexcept
  {
    return false;
  }

  template <class E>
  constexpr bool empty(std::initializer_list<E> il) noexcept
  {
    return il.size() == 0;
  }

  // from https://en.cppreference.com/w/cpp/iterator/data
  template <class C>
  constexpr auto data(C& c) -> decltype(c.data())
  {
    return c.data();
  }

  template <class C>
  constexpr auto data(const C& c) -> decltype(c.data())
  {
    return c.data();
  }

  template <class T, std::size_t N>
  constexpr T* data(T (&array)[N]) noexcept
  {
    return array;
  }

  template <class E>
  constexpr const E* data(std::initializer_list<E> il) noexcept
  {
    return il.begin();
  }

}

#endif

#endif  // HeterogeneousCore_CUDAUtilities_cuda_cxx17_h
