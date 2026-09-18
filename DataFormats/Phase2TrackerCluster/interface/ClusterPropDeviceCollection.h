//================================== alpaka device-side collection wrapper for ClusterPropSoA. ====================================== 
#ifndef DataFormats_Phase2TrackerCluster_interface_ClusterPropDeviceCollection_h
#define DataFormats_Phase2TrackerCluster_interface_ClusterPropDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/ClusterPropSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace Phase2RawToCluster {

    // SoA in device global memory
    using ClusterPropDeviceCollection = PortableCollection<::Phase2RawToCluster::ClusterPropSoA>;

  }  // namespace Phase2RawToCluster

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Phase2TrackerCluster_interface_ClusterPropDeviceCollection_h
