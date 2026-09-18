//================================== alpaka proxy collection providing unified Host/Device handling for ClusterPropSoA. ======================================
#ifndef DataFormats_Phase2TrackerCluster_interface_alpaka_ClusterPropSoACollection_h
#define DataFormats_Phase2TrackerCluster_interface_alpaka_ClusterPropSoACollection_h
#include "DataFormats/Phase2TrackerCluster/interface/ClusterPropDeviceCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/ClusterPropHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace Phase2RawToCluster {

  using ClusterPropSoACollection =
      std::conditional_t<std::is_same_v<ALPAKA_ACCELERATOR_NAMESPACE::Device, alpaka::DevCpu>, Phase2RawToCluster::ClusterPropHostCollection, ALPAKA_ACCELERATOR_NAMESPACE::Phase2RawToCluster::ClusterPropDeviceCollection>;

}  // namespace Phase2RawToCluster

#endif  // DataFormats_Phase2TrackerCluster_interface_alpaka_ClusterPropSoACollection_h
