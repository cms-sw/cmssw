#include "DataFormats/Provenance/interface/HardwareResourcesDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/compactStringSerializer.h"

#include <iterator>
#include <ostream>

namespace edm {
  HardwareResourcesDescription::HardwareResourcesDescription(std::string_view serialized) {
    // allowing empty input is mostly for backwards compatibility
    if (not serialized.empty()) {
      auto ret = edm::compactString::deserialize(serialized,
                                                 microarchitecture,
                                                 std::back_inserter(cpuModels),
                                                 std::back_inserter(selectedAccelerators),
                                                 std::back_inserter(gpuModels));
      // not comparing against serialized.size() to allow serialized
      // to have more content (for kind of forward compatibility)
      if (ret == 0) {
        throw Exception(errors::EventCorruption) << "Failed to deserialize HardwareResourcesDescription string format";
      }
    }
  }

  std::string HardwareResourcesDescription::serialize() const {
    return edm::compactString::serialize(microarchitecture, cpuModels, selectedAccelerators, gpuModels);
  }

  bool HardwareResourcesDescription::operator==(HardwareResourcesDescription const& other) const {
    return microarchitecture == other.microarchitecture and std::ranges::equal(cpuModels, other.cpuModels) and
           std::ranges::equal(selectedAccelerators, other.selectedAccelerators) and
           std::ranges::equal(gpuModels, other.gpuModels);
  }

  std::ostream& operator<<(std::ostream& os, HardwareResourcesDescription const& rd) {
    auto printContainer = [&os](std::string_view header, std::vector<std::string> const& cont) {
      os << header << ": " << cont.front();
      for (auto it = cont.begin() + 1; it != cont.end(); ++it) {
        os << ", " << *it;
      }
    };

    os << "uarch: " << rd.microarchitecture << "\n";
    if (not rd.cpuModels.empty()) {
      printContainer("CPU models", rd.cpuModels);
      os << "\n";
    }
    if (not rd.selectedAccelerators.empty()) {
      printContainer("Selected accelerators", rd.selectedAccelerators);
      os << "\n";
    }
    if (not rd.gpuModels.empty()) {
      printContainer("GPU models", rd.gpuModels);
    }
    return os;
  }
}  // namespace edm
