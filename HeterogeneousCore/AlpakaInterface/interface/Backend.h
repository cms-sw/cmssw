#ifndef HeterogeneousCore_AlpakaInterface_interface_Backend_h
#define HeterogeneousCore_AlpakaInterface_interface_Backend_h

#include <string_view>

namespace cms::alpakatools {
  // Enumeration whose value EDModules can put in the event
  enum class Backend : unsigned short { SerialSync = 0, CudaAsync = 1, ROCmAsync = 2, TbbAsync = 3, size };

  Backend toBackend(std::string_view name);
  std::string_view toString(Backend backend);
}  // namespace cms::alpakatools

#endif
