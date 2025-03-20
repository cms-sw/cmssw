#ifndef DataFormats_Provenance_interface_HardwareResourcesDescription_h
#define DataFormats_Provenance_interface_HardwareResourcesDescription_h

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace edm {
  /**
   * Description of various hardware details stored as part of
   * ProcessConfiguration
   *
   * An object can be serialized as a string. An object with empty
   * content will be serialized as an empty string (this was a simple
   * way to preserve backwards (and forwards) compatibility in
   * ProcessHistory for the ID checksum)
   */
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
