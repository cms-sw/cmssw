#pragma once

#include <typeinfo>
#include <type_traits>

#include <algorithm>

#include <string>
#include <bitset>
#include <utility>
#include <tuple>
#include <memory>
#include <array>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>

#include <cstddef>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"

namespace cond {
  namespace serialization {

    template <typename T>
    bool equal(const T& first, const T& second) {
      // This function takes advantage of template argument deduction,
      // making it easier to use than the access<T> template.

      // It is also called by the access<T>::equal_() methods themselves
      // if they need to compare objects. This means all comparisons
      // pass by here.

      // Therefore, we could easily first check here whether the address of
      // the objects is the same or add debugging code. In our use case,
      // however, most of the objects will have different addresses.
      return access<T>::equal_(first, second);
    }

    template <typename T>
    struct access<T, typename std::enable_if<std::is_integral<T>::value or std::is_enum<T>::value>::type> {
      static bool equal_(const T first, const T second) { return first == second; }
    };

    template <typename T>
    struct access<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
      static bool equal_(const T first, const T second) {
        // TODO: we consider all NaNs to be equal -- should we even allow to serialize them?
        if (std::isnan(first) or std::isnan(second))
          return std::isnan(first) and std::isnan(second);

        if (std::isinf(first) or std::isinf(second))
          return std::isinf(first) and std::isinf(second) and std::signbit(first) == std::signbit(second);

        // TODO: consider expected precision for cross-platform serialization
        return first == second;
      }
    };

    template <>
    struct access<std::string> {
      static bool equal_(const std::string& first, const std::string& second) { return first == second; }
    };

    template <std::size_t N>
    struct access<std::bitset<N>> {
      static bool equal_(const std::bitset<N>& first, const std::bitset<N>& second) { return first == second; }
    };

    template <typename T, typename U>
    struct access<std::pair<T, U>> {
      static bool equal_(const std::pair<T, U>& first, const std::pair<T, U>& second) {
        return equal(first.first, second.first) and equal(first.second, second.second);
      }
    };

    template <std::size_t N, typename... Ts>
    struct equal_tuple {
      static bool equal_(const std::tuple<Ts...>& first, const std::tuple<Ts...>& second) {
        if (not equal(std::get<N - 1>(first), std::get<N - 1>(second)))
          return false;

        return equal_tuple<N - 1, Ts...>::equal_(first, second);
      }
    };

    template <typename... Ts>
    struct equal_tuple<0, Ts...> {
      static bool equal_(const std::tuple<Ts...>& first, const std::tuple<Ts...>& second) { return true; }
    };

    template <typename... Ts>
    struct access<std::tuple<Ts...>> {
      static bool equal_(const std::tuple<Ts...>& first, const std::tuple<Ts...>& second) {
        return equal_tuple<sizeof...(Ts), Ts...>::equal_(first, second);
      }
    };

    template <typename T>
    struct access<T, typename std::enable_if<std::is_pointer<T>::value>::type> {
      static bool equal_(const T first, const T second) {
        if (first == nullptr or second == nullptr)
          return first == second;

        // Compare the addresses first -- even if equal() does not
        // do it for all types, if we are serializing pointers we may
        // have some use case of containers of pointers to a small
        // set of real objects.
        return first == second or equal(*first, *second);
      }
    };

#define equal_pointer(TYPE)                                                                                      \
  template <typename T>                                                                                          \
  struct access<TYPE<T>> {                                                                                       \
    static bool equal_(const TYPE<T>& first, const TYPE<T>& second) { return equal(first.get(), second.get()); } \
  };

    equal_pointer(std::unique_ptr);
    equal_pointer(std::shared_ptr);
#undef equal_pointer

    template <typename T, std::size_t N>
    struct access<T[N]> {
      static bool equal_(const T (&first)[N], const T (&second)[N]) {
        for (std::size_t i = 0; i < N; ++i)
          if (not equal(first[i], second[i]))
            return false;
        return true;
      }
    };

    template <typename T, std::size_t N>
    struct access<std::array<T, N>> {
      static bool equal_(const std::array<T, N>& first, const std::array<T, N>& second) {
        for (std::size_t i = 0; i < N; ++i)
          if (not equal(first[i], second[i]))
            return false;
        return true;
      }
    };

#define equal_sequence(TYPE)                                                                                           \
  template <typename T>                                                                                                \
  struct access<TYPE<T>> {                                                                                             \
    static bool equal_(const TYPE<T>& first, const TYPE<T>& second) {                                                  \
      return first.size() == second.size() &&                                                                          \
             std::equal(first.cbegin(),                                                                                \
                        first.cend(),                                                                                  \
                        second.cbegin(),                                                                               \
                        [](decltype(*first.cbegin()) a, decltype(*first.cbegin()) b) -> bool { return equal(a, b); }); \
    }                                                                                                                  \
  };

    equal_sequence(std::vector);
    equal_sequence(std::deque);
    equal_sequence(std::list);
    equal_sequence(std::set);       // ordered
    equal_sequence(std::multiset);  // ordered
#undef equal_sequence

    // forward_list is a sequence, but does not provide size() and we are not yet
    // in C++14 so we cannot use the 4 iterators version of std::equal()
    template <typename T>
    struct access<std::forward_list<T>> {
      static bool equal_(const std::forward_list<T>& first, const std::forward_list<T>& second) {
        auto first_it = first.cbegin();
        auto second_it = second.cbegin();

        while (first_it != first.cend() and second_it != second.cend()) {
          if (not equal(*first_it, *second_it))
            return false;
          first_it++;
          second_it++;
        }

        return first_it == first.cend() and second_it == second.cend();
      }
    };

// map is ordered too, we can iterate like a sequence
#define equal_mapping(TYPE)                                                                                            \
  template <typename T, typename U>                                                                                    \
  struct access<TYPE<T, U>> {                                                                                          \
    static bool equal_(const TYPE<T, U>& first, const TYPE<T, U>& second) {                                            \
      return first.size() == second.size() &&                                                                          \
             std::equal(first.cbegin(),                                                                                \
                        first.cend(),                                                                                  \
                        second.cbegin(),                                                                               \
                        [](decltype(*first.cbegin()) a, decltype(*first.cbegin()) b) -> bool { return equal(a, b); }); \
    }                                                                                                                  \
  };

    equal_mapping(std::map);
#undef equal_mapping

#define equal_unorderedmapping(TYPE)                                        \
  template <typename T, typename U>                                         \
  struct access<TYPE<T, U>> {                                               \
    static bool equal_(const TYPE<T, U>& first, const TYPE<T, U>& second) { \
      if (first.size() != second.size())                                    \
        return false;                                                       \
                                                                            \
      auto first_it = first.cbegin();                                       \
      while (first_it != first.cend()) {                                    \
        auto second_it = second.find(first_it->first);                      \
        if (second_it == second.cend())                                     \
          return false;                                                     \
        if (not equal(first_it->second, second_it->second))                 \
          return false;                                                     \
        first_it++;                                                         \
      }                                                                     \
      return true;                                                          \
    }                                                                       \
  };

    equal_unorderedmapping(std::unordered_map);
#undef equal_unorderedmapping

  }  // namespace serialization
}  // namespace cond
