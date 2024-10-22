#ifndef DataFormats_BeamSpot_interface_BeamSpotPOD_h
#define DataFormats_BeamSpot_interface_BeamSpotPOD_h

// This struct is a simplified representation of the beamspot data
// used as the underlying type for data transfers and operations in
// heterogeneous code (e.g. in CUDA code).

// The covariance matrix is not used in that code, so is left out here.

// align to the CUDA L1 cache line size
struct alignas(128) BeamSpotPOD {
  float x;  // position
  float y;
  float z;
  float sigmaZ;
  float beamWidthX;
  float beamWidthY;
  float dxdz;
  float dydz;
  float emittanceX;
  float emittanceY;
  float betaStar;
};

#endif  // DataFormats_BeamSpot_interface_BeamSpotPOD_h
