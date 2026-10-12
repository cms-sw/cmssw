#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <format>
#include <limits>
#include <stdexcept>
#include <typeinfo>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace portablecollection {

  template <std::size_t I = 0, typename TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::num_cols; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::num_cols) {
      const auto dest_size = std::get<I>(dest.buff).size();
      const auto src_size = std::get<I>(src.buff).size();
      assert(dest_size <= src_size && "portablecollection deepCopy: Destination buffer is larger than source buffer");

      if constexpr (ConstDescriptor::template isEigenColumn<I>()) {
        if (dest_size != src_size) {
          using EigenType = typename std::tuple_element_t<I, decltype(dest.parameterTypes)>::ValueType;
          const auto row_size = dest_size / (EigenType::RowsAtCompileTime * EigenType::ColsAtCompileTime);
          for (int i = 0; i < EigenType::RowsAtCompileTime * EigenType::ColsAtCompileTime; ++i) {
            const auto dest_offset = i * std::get<I>(dest.parameterTypes).stride();
            const auto src_offset = i * std::get<I>(src.parameterTypes).stride();

            alpaka::memcpy(
                queue,
                alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data() + dest_offset, row_size),
                alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data() + src_offset, row_size));
          }
        } else {
          alpaka::memcpy(queue,
                         alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), dest_size),
                         alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), dest_size));
        }
      } else {
        alpaka::memcpy(queue,
                       alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), dest_size),
                       alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), dest_size));
      }
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  // Helper function implementing the recursive deep copy for blocks
  template <std::size_t I = 0, typename TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::blocksNumber; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::blocksNumber) {
      deepCopy(queue, std::get<I>(dest.buff), std::get<I>(src.buff));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  template <std::integral Int>
  constexpr int size_cast(Int input) {
    if ((std::is_signed_v<Int> && input < 0) || input > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          std::format("Invalid input value for size of PortableCollection: cannot be narrowed to positive int32. "
                      "Source type: {}, value: {} ",
                      edm::typeDemangle(typeid(Int).name()),
                      input));
    }
    return static_cast<int>(input);
  }

}  // namespace portablecollection

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h
