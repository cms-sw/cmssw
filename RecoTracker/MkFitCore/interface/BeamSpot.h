#ifndef RecoTracker_MkFitCore_interface_BeamSpot_h
#define RecoTracker_MkFitCore_interface_BeamSpot_h

namespace mkfit {
  struct BeamSpot {
    float x = 0, y = 0, z = 0;
    float sigmaZ = 5;
    float beamWidthX = 5e-4, beamWidthY = 5e-4;
    float dxdz = 0, dydz = 0;

    BeamSpot() = default;
    BeamSpot(float ix, float iy, float iz, float is, float ibx, float iby, float idxdz, float idydz)
        : x(ix), y(iy), z(iz), sigmaZ(is), beamWidthX(ibx), beamWidthY(iby), dxdz(idxdz), dydz(idydz) {}
  };
}  // namespace mkfit

#endif
