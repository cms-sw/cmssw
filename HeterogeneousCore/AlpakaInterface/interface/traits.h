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

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  // is_platform

  template <typename T>
  using is_platform = alpaka::concepts::ImplementsConcept<alpaka::ConceptPltf, T>;

  template <typename T>
  inline constexpr bool is_platform_v = is_platform<T>::value;

  // is_device

  template <typename T>
  using is_device = alpaka::concepts::ImplementsConcept<alpaka::ConceptDev, T>;

  template <typename T>
  inline constexpr bool is_device_v = is_device<T>::value;

  // is_accelerator

  template <typename T>
  using is_accelerator = alpaka::concepts::ImplementsConcept<alpaka::ConceptAcc, T>;

  template <typename T>
  inline constexpr bool is_accelerator_v = is_accelerator<T>::value;

  // is_queue

  template <typename T>
  using is_queue = alpaka::concepts::ImplementsConcept<alpaka::ConceptQueue, T>;

  template <typename T>
  inline constexpr bool is_queue_v = is_queue<T>::value;

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_traits_h
