#ifndef DataFormats_SiStripDigiSoA_interface_SiStripDigiHost_h
#define DataFormats_SiStripDigiSoA_interface_SiStripDigiHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiSoA.h"

namespace sistrip {
  // SoA with SiStripClusters fields in host memory
  using SiStripDigiHost = PortableHostCollection<SiStripDigiSoA>;
}  // namespace sistrip

#endif
