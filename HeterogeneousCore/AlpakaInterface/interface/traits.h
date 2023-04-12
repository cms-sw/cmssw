#ifndef HeterogeneousCore_AlpakaInterface_interface_traits_h
#define HeterogeneousCore_AlpakaInterface_interface_traits_h

#include <type_traits>

#if __cplusplus >= 202002L
namespace cms {
  using std::is_bounded_array;
  using std::is_unbounded_array;
}  // namespace cms
#else
#include <boost/type_traits/is_bounded_array.hpp>
#include <boost/type_traits/is_unbounded_array.hpp>
namespace cms {
  using boost::is_bounded_array;
  using boost::is_unbounded_array;
}  // namespace cms
#endif

namespace cms {
  template <typename T>
  inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

  template <typename T>
  inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;
}  // namespace cms

#endif  // HeterogeneousCore_AlpakaInterface_interface_traits_h
