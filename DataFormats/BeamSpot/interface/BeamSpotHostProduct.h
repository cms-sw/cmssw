#ifndef DataFormats_BeamSpot_interface_BeamSpotHostProduct_h
#define DataFormats_BeamSpot_interface_BeamSpotHostProduct_h

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/Portable/interface/PortableHostProduct.h"

// simplified representation of the beamspot data, in host memory
using BeamSpotHostProduct = PortableHostProduct<BeamSpotPOD>;

#endif  // DataFormats_BeamSpot_interface_BeamSpotHostProduct_h
