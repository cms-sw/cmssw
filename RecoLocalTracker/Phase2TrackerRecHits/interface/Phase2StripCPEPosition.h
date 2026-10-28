#ifndef RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEPosition_h
#define RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEPosition_h

#include <cstdint>

// The cluster position of Phase2StripCPE (RectangularPixelPhase2Topology::localX / localY) as constexpr functions for
// device code; testPhase2StripCPEPosition checks them against the topology.
namespace phase2StripCPE {

  // Per OT module: the RectangularPixelPhase2Topology constants used by localX and localY, and the Lorentz drift.
  struct ModuleParams {
    float pitchX;
    float pitchY;
    float bigPitchX;  // RectangularPixelPhase2Topology::pitchbigpixelX()
    float bigPitchY;
    float xOffset;  // RectangularPixelPhase2Topology::xoffset()
    float yOffset;
    int32_t nRows;
    int32_t nColumns;
    int32_t rowsPerRoc;
    int32_t columnsPerRoc;
    float coveredStrips;  // drift of the charge across the sensor thickness, in units of pitchX
  };

  struct LocalPosition {
    float x;
    float y;
  };

  // RectangularPixelPhase2Topology::localX
  constexpr float localX(ModuleParams const& params, float mpx) {
    int binoffx = int(mpx);
    float fractionX = mpx - float(binoffx);
    float localPitchX = params.pitchX;
    int secondHalfX = 0;
    if (binoffx >= (params.nRows / 2 - 2 + 2 * params.nRows / params.rowsPerRoc)) {
      binoffx = binoffx - 2 * params.nRows / params.rowsPerRoc;
      secondHalfX = 1;
    } else if (((params.nRows / 2 - 2) <= binoffx) &&
               (binoffx < (params.nRows / 2 - 2 + 2 * params.nRows / params.rowsPerRoc))) {
      binoffx = params.nRows / 2 - 2;
      fractionX = mpx - float(params.nRows / 2 - 2);
      localPitchX = params.bigPitchX;
    }
    return float(binoffx * params.pitchX) + fractionX * localPitchX +
           secondHalfX * 2 * params.bigPitchX * params.nRows / params.rowsPerRoc + params.xOffset;
  }

  // RectangularPixelPhase2Topology::localY
  constexpr float localY(ModuleParams const& params, float mpy) {
    int binoffy = int(mpy);
    float fractionY = mpy - float(binoffy);
    float localPitchY = params.pitchY;
    int secondHalfY = 0;
    if (binoffy >= (params.nColumns / 2 - 1 + params.nColumns / params.columnsPerRoc)) {
      binoffy = binoffy - params.nColumns / params.columnsPerRoc;
      secondHalfY = 1;
    } else if (((params.nColumns / 2 - 1) <= binoffy) &&
               (binoffy < (params.nColumns / 2 - 1 + params.nColumns / params.columnsPerRoc))) {
      binoffy = params.nColumns / 2 - 1;
      fractionY = mpy - float(params.nColumns / 2 - 1);
      localPitchY = params.bigPitchY;
    }
    return float(binoffy * params.pitchY) + fractionY * localPitchY +
           secondHalfY * params.bigPitchY * params.nColumns / params.columnsPerRoc + params.yOffset;
  }

  // Phase2StripCPE::localParameters: the cluster centre (Phase2TrackerCluster1D::center()) shifted back by half the
  // Lorentz drift, and the middle of the column
  constexpr LocalPosition localPosition(ModuleParams const& params,
                                        uint32_t firstStrip,
                                        uint32_t column,
                                        uint16_t size) {
    const float center = float(firstStrip) + 0.5f * size;
    const float ix = center - 0.5f * params.coveredStrips;
    const float iy = float(column) + 0.5f;
    return {localX(params, ix), localY(params, iy)};
  }

}  // namespace phase2StripCPE

#endif  // RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEPosition_h
