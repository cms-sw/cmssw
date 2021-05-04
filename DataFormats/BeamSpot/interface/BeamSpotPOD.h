#ifndef DataFormats_BeamSpot_interface_BeamSpotPOD_h
#define DataFormats_BeamSpot_interface_BeamSpotPOD_h

// This struct is a transient-only, simplified representation of the beamspot
// data used as the underlying type for data transfers and operations in
// heterogeneous code (e.g. in CUDA code).

// The covariance matrix is not used in that code, so is left out here.

// align to the CUDA L1 cache line size
struct alignas(128) BeamSpotPOD {
  float x, y, z;  // position
  float sigmaZ;
  float beamWidthX, beamWidthY;
  float dxdz, dydz;
  float emittanceX, emittanceY;
  float betaStar;
};

#endif  // DataFormats_BeamSpot_interface_BeamSpotPOD_h
