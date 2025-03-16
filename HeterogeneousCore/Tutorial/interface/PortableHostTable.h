#ifndef HeterogeneousCore_Tutorial_interface_PortableHostTable_h
#define HeterogeneousCore_Tutorial_interface_PortableHostTable_h

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

  class PortableHostTable {
  public:
    using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
    using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;

    PortableHostTable(alpaka_common::DevHost const& host, int x_bins, int y_bins)
        : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Table::size_bytes(x_bins, y_bins))},
          table_{x_bins, y_bins, std::span<std::byte>(buffer_.data(), Table::size_bytes(x_bins, y_bins))}  //
    {
      assert(reinterpret_cast<uintptr_t>(buffer_.data()) % alignof(float) == 0);
    }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    PortableHostTable(TQueue const& queue, int x_bins, int y_bins)
        : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Table::size_bytes(x_bins, y_bins))},
          table_{x_bins, y_bins, std::span<std::byte>(buffer_.data(), Table::size_bytes(x_bins, y_bins))}  //
    {
      assert(reinterpret_cast<uintptr_t>(buffer_.data()) % alignof(float) == 0);
    }

    int x_bins() const { return table_.x_bins(); }

    int y_bins() const { return table_.y_bins(); }

    std::span<const float> get_x_axis() const { return table_.get_x_axis(); }

    std::span<const float> get_y_axis() const { return table_.get_y_axis(); }

    std::span<const float> get_data() const { return table_.get_data(); }

    void set_x_axis(std::span<const float> const& axis) { table_.set_x_axis(axis); }

    void set_y_axis(std::span<const float> const& axis) { table_.set_y_axis(axis); }

    void set_data(std::span<const float> const& data) { table_.set_data(data); }

    float get(float x, float y) const { return table_.get(x, y); }

    void set(float x, float y, float value) { table_.set(x, y, value); }

    Table& table() { return table_; }
    Table const& table() const { return table_; }

    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }

  private:
    Buffer buffer_;
    Table table_;
  };

}  // namespace tutorial

#endif  // HeterogeneousCore_Tutorial_interface_PortableHostTable_h
