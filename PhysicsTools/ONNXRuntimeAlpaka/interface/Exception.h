#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_Exception_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_Exception_h

#include <string>

namespace cms::Ort::alpakatools::detail {

  // Throw a cms::Exception; implemented in a source file to avoid replicating the exception handling code in every
  // translation unit that includes the headers of this package.
  [[noreturn]] void throwException(const std::string& category, const std::string& message);

  // Throw a cms::Exception, with the given context.
  [[noreturn]] void throwException(const std::string& category, const std::string& message, const std::string& context);

}  // namespace cms::Ort::alpakatools::detail

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_Exception_h
