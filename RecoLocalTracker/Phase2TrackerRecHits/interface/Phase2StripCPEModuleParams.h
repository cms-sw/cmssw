#ifndef RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEModuleParams_h
#define RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEModuleParams_h

#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

namespace phase2StripCPE {

  // The parameters of localPosition for a module with this topology and Lorentz drift (Phase2StripCPE::Param)
  inline ModuleParams moduleParams(RectangularPixelPhase2Topology const& topology, float coveredStrips) {
    return {topology.pitch().first,
            topology.pitch().second,
            topology.pitchbigpixelX(),
            topology.pitchbigpixelY(),
            topology.xoffset(),
            topology.yoffset(),
            topology.nrows(),
            topology.ncolumns(),
            topology.rowsperroc(),
            topology.colsperroc(),
            coveredStrips};
  }

}  // namespace phase2StripCPE

#endif  // RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEModuleParams_h
