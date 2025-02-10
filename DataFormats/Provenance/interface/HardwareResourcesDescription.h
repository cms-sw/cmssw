#ifndef DataFormats_Provenance_interface_HardwareResourcesDescription_h
#define DataFormats_Provenance_interface_HardwareResourcesDescription_h

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace edm {
  struct HardwareResourcesDescription {
    HardwareResourcesDescription() = default;
    explicit HardwareResourcesDescription(std::string_view serialized);

    std::string serialize() const;

    bool operator==(HardwareResourcesDescription const& other) const;

    std::string microarchitecture;
    std::vector<std::string> cpuModels;
    std::vector<std::string> selectedAccelerators;
    std::vector<std::string> gpuModels;
  };

  std::ostream& operator<<(std::ostream& os, HardwareResourcesDescription const& rd);
}  // namespace edm

#endif
