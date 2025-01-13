#ifndef RecoLocalTracker_ClusterParameterEstimator_interface_FrameSoADevice_h
#define RecoLocalTracker_ClusterParameterEstimator_interface_FrameSoADevice_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoALayout.h"

template <typename TDev>
using FrameSoADevice = PortableDeviceCollection<FrameSoALayout, TDev>;

#endif  // RecoLocalTracker_ClusterParameterEstimator_interface_FrameSoADevice_h
