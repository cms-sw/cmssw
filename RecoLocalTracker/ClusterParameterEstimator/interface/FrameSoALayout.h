#ifndef RecoLocalTracker_ClusterParameterEstimator_interface_FrameLayout_h
#define RecoLocalTracker_ClusterParameterEstimator_interface_FrameLayout_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(FrameLayout, SOA_COLUMN(SOAFrame<float>, detFrame))

using FrameSoALayout = FrameLayout<>;
using FrameSoAView = FrameSoALayout::View;
using FrameSoAConstView = FrameSoALayout::ConstView;

#endif  //RecoLocalTracker_ClusterParameterEstimator_interface_FrameLayout_h
