#ifndef HeterogeneousCore_Tutorial_interface_PortableTable_h
#define HeterogeneousCore_Tutorial_interface_PortableTable_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"
#include "HeterogeneousCore/Tutorial/interface/PortableDeviceTable.h"
#include "HeterogeneousCore/Tutorial/interface/PortableHostTable.h"

namespace tutorial::traits {

  // trait for a generic struct-based product
  template <typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableTableTrait {
    using type = ::tutorial::PortableDeviceTable<TDev>;
  };

  // specialise for host device
  template <>
  struct PortableTableTrait<alpaka_common::DevHost> {
    using type = ::tutorial::PortableHostTable;
  };

}  // namespace tutorial::traits

namespace tutorial {

  // type alias for a generic struct-based product
  template <typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  using PortableTable = typename ::tutorial::traits::PortableTableTrait<TDev>::type;

}  // namespace tutorial

// define how to copy PortableTable between host and device
namespace cms::alpakatools {

  template <typename TDev>
    requires alpaka::isDevice<TDev>
  struct CopyToHost<tutorial::PortableDeviceTable<TDev>> {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    static auto copyAsync(TQueue& queue, tutorial::PortableDeviceTable<TDev> const& src) {
      tutorial::PortableHostTable dst(queue, src.x_bins(), src.y_bins());
      alpaka::memcpy(queue, dst.buffer(), src.buffer());
      return dst;
    }
  };

  template <>
  struct CopyToDevice<tutorial::PortableHostTable> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, tutorial::PortableHostTable const& src) {
      tutorial::PortableDeviceTable<alpaka::Dev<TQueue>> dst(queue, src.x_bins(), src.y_bins());
      alpaka::memcpy(queue, dst.buffer(), src.buffer());
      return dst;
    }
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_Tutorial_interface_PortableTable_h
