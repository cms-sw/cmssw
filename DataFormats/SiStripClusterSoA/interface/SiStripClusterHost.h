#ifndef DataFormats_SiStripClusterSoA_interface_SiStripClusterHost_h
#define DataFormats_SiStripClusterSoA_interface_SiStripClusterHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClusterSoA.h"

namespace sistrip {
  // SoA with SiStripCluster fields in host memory
  using SiStripClusterHost = PortableHostCollection<SiStripClusterSoA>;
}  // namespace sistrip

#endif  // DataFormats_SiStripClusterSoA_interface_SiStripClusterHost_h
