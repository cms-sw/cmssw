#ifndef DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h
#define DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h

#include "DataFormats/BeamSpot/interface/BeamSpotHost.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // simplified representation of the beamspot data, in device global memory
  using BeamSpotDevice = PortableObject<BeamSpotPOD>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(BeamSpotDevice, BeamSpotHost);

#endif  // DataFormats_BeamSpot_interface_alpaka_BeamSpotDevice_h
