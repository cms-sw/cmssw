#include <format>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoACommon.h"

namespace cms::soa::detail {
  [[noreturn]] void throwRuntimeError(const char* message) { throw std::runtime_error(message); }

  template <>
  [[noreturn]] void throwOutOfRangeError<RangeChecking::extended>(
      const char* message, const IndexWithSourceLocation<RangeChecking::extended>& index, cms::soa::size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {} at file {} at line {}\n",
                                        message,
                                        index.value_,
                                        range,
                                        index.location_.file_name(),
                                        index.location_.line()));
  }

  template <>
  [[noreturn]] void throwOutOfRangeError<RangeChecking::enabled>(
      const char* message, const IndexWithSourceLocation<RangeChecking::enabled>& index, cms::soa::size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {}\n", message, index.value_, range));
  }

  template <>
  [[noreturn]] void throwOutOfRangeError<RangeChecking::disabled>(
      const char* message, const IndexWithSourceLocation<RangeChecking::disabled>& index, cms::soa::size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {}\n", message, index.value_, range));
  }
}  // namespace cms::soa::detail
