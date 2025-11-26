#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_Environment_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_Environment_h

#include <compare>
#include <cstdint>

namespace torchtest {

  enum class Environment : int { kProduction = 0, kDevelopment = 1, kTest = 2, kDebug = 3 };

  constexpr std::strong_ordering operator<=>(Environment t, Environment u) {
    return static_cast<int>(t) <=> static_cast<int>(u);
  }

}  // namespace torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_Environment_h
