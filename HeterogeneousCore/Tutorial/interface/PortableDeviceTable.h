#ifndef HeterogeneousCore_Tutorial_interface_PortableDeviceTable_h
#define HeterogeneousCore_Tutorial_interface_PortableDeviceTable_h

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/Tutorial/interface/Table.h"

namespace tutorial {

  template <typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class PortableDeviceTable {
    static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                  "Use PortableHostTable instead of PortableDeviceTable<DevHost>");

  public:
    using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
    using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;

    PortableDeviceTable(TDev const& device, int x_bins, int y_bins)
        : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Table::size_bytes(x_bins, y_bins))},
          table_{x_bins, y_bins, std::span<std::byte>(buffer_.data(), Table::size_bytes(x_bins, y_bins))}  //
    {
      assert(reinterpret_cast<uintptr_t>(buffer_.data()) % alignof(float) == 0);
    }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    PortableDeviceTable(TQueue const& queue, int x_bins, int y_bins)
        : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Table::size_bytes(x_bins, y_bins))},
          table_{x_bins, y_bins, std::span<std::byte>(buffer_.data(), Table::size_bytes(x_bins, y_bins))}  //
    {
      assert(reinterpret_cast<uintptr_t>(buffer_.data()) % alignof(float) == 0);
    }

    int x_bins() const { return table_.x_bins(); }

    int y_bins() const { return table_.y_bins(); }

    Table& table() { return table_; }
    Table const& table() const { return table_; }

    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }

  private:
    Buffer buffer_;
    Table table_;
  };

}  // namespace tutorial

#endif  // HeterogeneousCore_Tutorial_interface_PortableDeviceTable_h
