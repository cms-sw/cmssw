#ifndef DataFormats_Provenance_interface_ResourcesDescription_h
#define DataFormats_Provenance_interface_ResourcesDescription_h

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace edm {
  struct ResourcesDescription {
    ResourcesDescription() = default;
    explicit ResourcesDescription(std::string_view serialized);

    std::string serialize() const;

    bool operator==(ResourcesDescription const& other) const;

    std::string microarchitecture;
    std::vector<std::string> cpuModels;
    std::vector<std::string> selectedAccelerators;
    std::vector<std::string> gpuModels;
  };

  std::ostream& operator<<(std::ostream& os, ResourcesDescription const& rd);
}  // namespace edm

#endif
