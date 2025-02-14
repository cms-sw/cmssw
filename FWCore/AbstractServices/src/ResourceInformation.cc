#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/Utilities/interface/stringize.h"

namespace edm {
  ResourceInformation::ResourceInformation() = default;
  ResourceInformation::~ResourceInformation() = default;

  HardwareResourcesDescription ResourceInformation::hardwareResourcesDescription() const {
    HardwareResourcesDescription ret;
    ret.microarchitecture = EDM_STRINGIZE(CMS_MICRO_ARCH);
    ret.cpuModels = cpuModels();
    ret.selectedAccelerators = selectedAccelerators();
    ret.gpuModels = gpuModels();
    return ret;
  }
}  // namespace edm
