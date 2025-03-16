#ifndef HeterogeneousCore_Tutorial_interface_Table_h
#define HeterogeneousCore_Tutorial_interface_Table_h

#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>

#include <alpaka/alpaka.hpp>

namespace tutorial {

  class Table {
  public:
    Table(int x_bins, int y_bins, std::span<std::byte> buffer)
        : x_bins_(x_bins),
          y_bins_(y_bins),
          x_axis_(reinterpret_cast<float*>(buffer.data())),
          y_axis_(x_axis_ + x_bins_ + 1),
          data_(y_axis_ + y_bins_ + 1)  //
    {
      assert(buffer.size_bytes() >= size_bytes());
    }

    ALPAKA_FN_HOST_ACC
    int x_bins() const { return x_bins_; }

    ALPAKA_FN_HOST_ACC
    int y_bins() const { return y_bins_; }

    ALPAKA_FN_HOST_ACC
    std::span<const float> get_x_axis() const  //
    {
      return std::span<const float>(x_axis_, x_bins_ + 1);
    }

    ALPAKA_FN_HOST_ACC
    void set_x_axis(std::span<const float> const& axis) {
      assert(static_cast<int>(axis.size()) == x_bins_ + 1);
      std::memcpy(x_axis_, axis.data(), axis.size_bytes());
    }

    ALPAKA_FN_HOST_ACC
    std::span<const float> get_y_axis() const  //
    {
      return std::span<const float>(y_axis_, y_bins_ + 1);
    }

    ALPAKA_FN_HOST_ACC
    void set_y_axis(std::span<const float> const& axis) {
      assert(static_cast<int>(axis.size()) == y_bins_ + 1);
      std::memcpy(y_axis_, axis.data(), axis.size_bytes());
    }

    ALPAKA_FN_HOST_ACC
    std::span<const float> get_data() const { return std::span<const float>(data_, x_bins_ * y_bins_); }

    ALPAKA_FN_HOST_ACC
    void set_data(std::span<const float> const& data) {
      assert(static_cast<int>(data.size()) == x_bins_ * y_bins_);
      std::memcpy(data_, data.data(), data.size_bytes());
    }

    ALPAKA_FN_HOST_ACC
    float get(float x, float y) const {
      float const* ptr = bin(x, y);
      return (ptr == nullptr) ? 0.f : *ptr;
    }

    ALPAKA_FN_HOST_ACC
    void set(float x, float y, float value) {
      float* ptr = bin(x, y);
      if (ptr != nullptr)
        *ptr = value;
    }

    size_t size_bytes() const { return Table::size_bytes(x_bins_, y_bins_); }

    static constexpr size_t size_bytes(int x_bins, int y_bins) {
      return (x_bins + 1) * sizeof(float) +      // memory for the X axis
             (y_bins + 1) * sizeof(float) +      // memory for the Y axis
             (x_bins * y_bins) * sizeof(float);  // memory for the data
    }

  private:
    ALPAKA_FN_HOST_ACC
    float* bin(float x, float y) {
      // check for under and overflow
      if (x < x_axis_[0] or x >= x_axis_[x_bins_]) {
        return nullptr;
      }
      // linear search
      int i = 0;
      while (x >= x_axis_[i + 1]) {
        ++i;
      }
      assert(i < x_bins_);

      // check for under and overflow
      if (y < y_axis_[0] or y >= y_axis_[y_bins_]) {
        return nullptr;
      }
      // linear search
      int j = 0;
      while (y >= y_axis_[j + 1]) {
        ++j;
      }
      assert(j < y_bins_);

      return data_ + (j * x_bins_) + i;
    }

    ALPAKA_FN_HOST_ACC
    float const* bin(float x, float y) const { return const_cast<Table*>(this)->bin(x, y); }

    int x_bins_;
    int y_bins_;

    float* x_axis_;
    float* y_axis_;
    float* data_;
  };

}  // namespace tutorial

#endif  // HeterogeneousCore_Tutorial_interface_Table_h
