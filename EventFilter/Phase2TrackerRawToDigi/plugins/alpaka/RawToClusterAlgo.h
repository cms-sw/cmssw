//====================== Header with helper functions and shared algorithmic components used by the unpacker. ==========================
#ifndef EventFilter_Phase2TrackerRawToDigi_RawToClusterAlgo_h
#define EventFilter_Phase2TrackerRawToDigi_RawToClusterAlgo_h

#include "DataFormats/Phase2TrackerCluster/interface/ClusterPropDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

   void launchUnpacker(
    Queue& queue,
    cms::alpakatools::device_buffer<Device, unsigned char[]> const& rawdatabuff,
    cms::alpakatools::device_buffer<Device, size_t[]>        const& sizedatabuff,
    cms::alpakatools::device_buffer<Device, size_t[]>        const& offsetdatabuff,
    cms::alpakatools::device_buffer<Device, int[]>           const& detIdxModuleTypeDevice,
    cms::alpakatools::device_buffer<Device, uint32_t[]>      const& innerDetIdDevice,
    cms::alpakatools::device_buffer<Device, uint32_t[]>      const& outerDetIdDevice,
    Phase2RawToCluster::ClusterPropDeviceCollection::View out,
    uint32_t* globalCounter) ;

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // EventFilter_Phase2TrackerRawToDigi_RawToClusterAlgo_h
