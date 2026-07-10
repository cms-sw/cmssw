#include <format>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoACommon.h"

namespace cms::soa::detail {
  [[noreturn]] void throwRuntimeError(const char* message) { throw std::runtime_error(message); }

  [[noreturn]] void throwOutOfRangeError(const char* message, cms::soa::size_type index, cms::soa::size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {}", message, index, range));
  }
}  // namespace cms::soa::detail
