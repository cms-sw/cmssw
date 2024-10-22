#ifndef L1Trigger_TrackFindingTMTT_TrackerModule_h
#define L1Trigger_TrackFindingTMTT_TrackerModule_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <vector>
#include <set>
#include <array>
#include <map>
#include <cmath>

class TrackerGeometry;
class TrackerTopology;
class PixelGeomDetUnit;
class PixelTopology;

namespace tmtt {

  //=== Get info about tracker module

  class TrackerModule {
  public:
    enum BarrelModuleType { tiltedMinusZ = 1, tiltedPlusZ = 2, flat = 3 };

    // Info used to define firmware module type.
    struct ModuleTypeCfg {
      std::vector<double> pitchVsType;
      std::vector<double> spaceVsType;
      std::vector<bool> barrelVsType;
      std::vector<bool> psVsType;
      std::vector<bool> tiltedVsType;
    };

    // Here detId is ID of lower sensor in stacked module.
    TrackerModule(const TrackerGeometry* trackerGeometry,
                  const TrackerTopology* trackerTopology,
                  const ModuleTypeCfg& moduleTypeCfg,
                  const DetId& detId);

    // Det ID of lower sensor in stacked module.
    const DetId& detId() const { return detId_; }
    unsigned int rawDetId() const { return detId_.rawId(); }
    // Det ID of stacked module.
    const DetId& stackedDetId() const { return stackedDetId_; }
    unsigned int rawStackedDetId() const { return stackedDetId_.rawId(); }
    // Tracker specific DetUnit & topology.
    const PixelGeomDetUnit* specDet() const { return specDet_; }
    const PixelTopology* specTopol() const { return specTopol_; }
    // Coordinates of centre of two sensors in (r,phi,z)
    float minR() const { return moduleMinR_; }
    float maxR() const { return moduleMaxR_; }
    float minPhi() const { return moduleMinPhi_; }
    float maxPhi() const { return moduleMaxPhi_; }
    float minZ() const { return moduleMinZ_; }
    float maxZ() const { return moduleMaxZ_; }
    // Polar angle of module.
    float theta() const { return atan2(moduleMinR_, moduleMinZ_); }
    // Which of two sensors in module is furthest from beam-line?
    bool outerModuleAtSmallerR() const { return outerModuleAtSmallerR_; }
    // Module type: PS or 2S?
    bool psModule() const { return psModule_; }
    bool barrel() const { return barrel_; }
    // Tracker layer ID number (1-6 = barrel layer; 11-15 = endcap A disk; 21-25 = endcap B disk)
    unsigned int layerId() const { return layerId_; }
    // Reduced layer ID (in range 1-7), for  packing into 3 bits to simplify the firmware.
    unsigned int layerIdReduced() const { return layerIdReduced_; }
    // Endcap ring of module (returns zero in case of barrel)
    unsigned int endcapRing() const { return endcapRing_; }
    // True if stub is in tilted barrel module.
    bool tiltedBarrel() const { return tiltedBarrel_; }
    // Angle between normal to module and beam-line along +ve z axis. (In range -PI/2 to +PI/2).
    float tiltAngle() const { return tiltAngle_; }
    // Width of sensitive region of sensor.
    float sensorWidth() const { return sensorWidth_; }
    // Sensor spacing in module
    float sensorSpacing() const { return sensorSpacing_; }
    // No. of strips in sensor.
    unsigned int nStrips() const { return nStrips_; }
    // Strip pitch (or pixel pitch along shortest axis).
    float stripPitch() const { return stripPitch_; }
    // Strip length (or pixel pitch along longest axis).
    float stripLength() const { return stripLength_; }
    // Hit resolution perpendicular to strip (or to longest pixel axis). Measures phi.
    float sigmaPerp() const { return invRoot12 * stripPitch_; }
    // Hit resolution parallel to strip (or to longest pixel axis). Measures r or z.
    float sigmaPar() const { return invRoot12 * stripLength_; }
    // Sensor pitch over separation.
    float pitchOverSep() const { return stripPitch_ / sensorSpacing_; }
    // "B" parameter correction for module tilt.
    float paramB() const { return std::abs(cos(theta() - tiltAngle()) / sin(theta())); }
    // Module type ID defined by firmware.
    unsigned int moduleTypeID() const { return moduleTypeID_; }

    //--- Utilties

    // Calculate reduced layer ID (in range 1-7), for  packing into 3 bits to simplify the firmware.
    static unsigned int calcLayerIdReduced(unsigned int layerId) {
      // Don't bother distinguishing two endcaps, as no track can have stubs in both.
      unsigned int lay = (layerId < 20) ? layerId : layerId - 10;

      // No genuine track can have stubs in both barrel layer 6 and endcap disk 11 etc., so merge their layer IDs.
      if (lay == 6)
        lay = 11;
      else if (lay == 5)
        lay = 12;
      else if (lay == 4)
        lay = 13;
      else if (lay == 3)
        lay = 15;
      // At this point, the reduced layer ID can have values of 1, 2, 11, 12, 13, 14, 15. So correct to put in range 1-7.
      if (lay > 10)
        lay -= 8;

      if (lay < 1 || lay > 7)
        throw cms::Exception("LogicError") << "TrackerModule: Reduced layer ID out of expected range";

      return lay;
    }

    // Get module type ID defined by firmware.
    unsigned int calcModuleType(float pitch, float space, bool barrel, bool tiltedBarrel, bool psModule) const;

  private:
    DetId detId_;
    DetId stackedDetId_;
    const PixelGeomDetUnit* specDet_;
    const PixelTopology* specTopol_;
    float moduleMinR_;
    float moduleMaxR_;
    float moduleMinPhi_;
    float moduleMaxPhi_;
    float moduleMinZ_;
    float moduleMaxZ_;
    bool outerModuleAtSmallerR_;
    bool psModule_;
    bool barrel_;
    unsigned int layerId_;
    unsigned int layerIdReduced_;
    unsigned int endcapRing_;
    bool tiltedBarrel_;
    float tiltAngle_;
    float sensorWidth_;
    float sensorSpacing_;
    unsigned int nStrips_;
    float stripPitch_;
    float stripLength_;
    unsigned int moduleTypeID_;

    ModuleTypeCfg moduleTypeCfg_;

    static const float invRoot12;
  };

}  // namespace tmtt
#endif
