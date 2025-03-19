#ifndef DataFormats_SiStripClusterSoA_interface_SiStripClustersHost_h
#define DataFormats_SiStripClusterSoA_interface_SiStripClustersHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"

namespace sistrip {

  // SoA with SiStripClusters fields in host memory
  using SiStripClustersHost = PortableHostCollection<SiStripClustersSoA>;

}  // namespace sistrip

#endif  // DataFormats_SiStripClusterSoA_interface_SiStripClustersHost_h
