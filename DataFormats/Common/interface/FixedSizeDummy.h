#ifndef MYMODULE_FixedSizeDummy_H
#define MYMODULE_FixedSizeDummy_H

#include <cstdint>
#include <vector>
#include <cstddef>

namespace edm {
  class FixedSizeDummy {
  public:
    FixedSizeDummy() = default;

    explicit FixedSizeDummy(size_t size) : data_(size, 123) {
      // Fills with 123 for easy validation
    }

    size_t size() const { return data_.size(); }

    const std::vector<uint8_t>& data() const { return data_; }
    std::vector<uint8_t>& data() { return data_; }

  private:
    std::vector<uint8_t> data_;  // ROOT knows how to stream this
  };
}

#endif
