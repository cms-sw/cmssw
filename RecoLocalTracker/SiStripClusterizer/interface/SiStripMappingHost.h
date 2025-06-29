#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingHost_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingSoA.h"

namespace sistrip {
  using SiStripMappingHost = PortableHostCollection<SiStripMappingSoA>;
}

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingHost_h
