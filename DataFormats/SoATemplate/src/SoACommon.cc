#include <format>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoACommon.h"

namespace cms::soa::detail {
  [[noreturn]] void throwRuntimeError(const char* message) { throw std::runtime_error(message); }

  [[noreturn]] void throwOutOfRangeError(const char* message,
                                         IndexWithSourceLocation index,
                                         cms::soa::size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {} at file {} at line {} \n",
                                        message,
                                        index.value_,
                                        range,
                                        index.location_.file_name(),
                                        index.location_.line()));
  }
}  // namespace cms::soa::detail
