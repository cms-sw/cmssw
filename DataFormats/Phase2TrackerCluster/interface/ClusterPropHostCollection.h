//================================== alpaka host-side collection wrapper for ClusterPropSoA. ==========================================
#ifndef DataFormats_Phase2TrackerCluster_interface_ClusterPropHostCollection_h
#define DataFormats_Phase2TrackerCluster_interface_ClusterPropHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/ClusterPropSoA.h"

namespace Phase2RawToCluster {

  // SoA with x, y, z, id fields in host memory
  using ClusterPropHostCollection = PortableHostCollection<ClusterPropSoA>;

}  // namespace Phase2RawToCluster

#endif  // DataFormats_Phase2TrackerCluster_interface_ClusterPropHostCollection_h
