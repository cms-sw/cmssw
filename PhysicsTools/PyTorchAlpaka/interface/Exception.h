#ifndef PhysicsTools_PyTorchAlpaka_interface_Exception_h
#define PhysicsTools_PyTorchAlpaka_interface_Exception_h

#include <string>

namespace cms::torch::alpakatools::detail {

  [[noreturn]] void throwException(const std::string& category, const std::string& message);

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_Exception_h
