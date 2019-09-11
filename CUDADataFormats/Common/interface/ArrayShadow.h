#ifndef CUDADataFormatsCommonArrayShadow_H
#define CUDADataFormatsCommonArrayShadow_H
#include <array>

template <typename A>
struct ArrayShadow {
  using T = typename A::value_type;
  constexpr static auto size() { return std::tuple_size<A>::value; }
  T data[std::tuple_size<A>::value];
};

#endif
