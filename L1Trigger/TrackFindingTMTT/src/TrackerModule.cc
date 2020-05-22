#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"

#include <iostream>
#include <sstream>
#include <mutex>

using namespace std;

namespace tmtt {

  namespace {
    std::once_flag printOnce;
  }

  const float TrackerModule::invRoot12 = sqrt(1. / 12.);

  //=== Get info about tracker module (where detId is ID of lower sensor in stacked module).

  TrackerModule::TrackerModule(const TrackerGeometry* trackerGeometry,
                               const TrackerTopology* trackerTopology,
                               const ModuleTypeCfg& moduleTypeCfg,
                               const DetId& detId)
      : moduleTypeCfg_(moduleTypeCfg) {
    detId_ = detId;                                 // Det ID of lower sensor in stacked module.
    stackedDetId_ = trackerTopology->stack(detId);  // Det ID of stacked module.

    // Get min & max (r,phi,z) coordinates of the centre of the two sensors containing this stub.
    const GeomDetUnit* det0 = trackerGeometry->idToDetUnit(detId);
    const GeomDetUnit* det1 = trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId));
    specDet_ = dynamic_cast<const PixelGeomDetUnit*>(det0);
    specTopol_ = dynamic_cast<const PixelTopology*>(&(specDet_->specificTopology()));

    float R0 = det0->position().perp();
    float R1 = det1->position().perp();
    float PHI0 = det0->position().phi();
    float PHI1 = det1->position().phi();
    float Z0 = det0->position().z();
    float Z1 = det1->position().z();
    moduleMinR_ = std::min(R0, R1);
    moduleMaxR_ = std::max(R0, R1);
    moduleMinPhi_ = std::min(PHI0, PHI1);
    moduleMaxPhi_ = std::max(PHI0, PHI1);
    moduleMinZ_ = std::min(Z0, Z1);
    moduleMaxZ_ = std::max(Z0, Z1);

    // Note if modules are flipped back-to-front.
    outerModuleAtSmallerR_ = (det0->position().mag() > det1->position().mag());

    // Note if module is PS or 2S, and whether in barrel or endcap.
    // From Geometry/TrackerGeometryBuilder/README.md
    psModule_ = (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP);
    barrel_ = detId.subdetId() == StripSubdetector::TOB || detId.subdetId() == StripSubdetector::TIB;

    // Encode layer ID (barrel layers: 1-6, endcap disks: 11-15 + 21-25)
    if (barrel_) {
      layerId_ = trackerTopology->layer(detId);  // barrel layer 1-6 encoded as 1-6
    } else {
      layerId_ = 10 * trackerTopology->side(detId) + trackerTopology->tidWheel(detId);
    }
    // Get reduced layer ID (in range 1-7), requiring only 3 bits for firmware.
    layerIdReduced_ = TrackerModule::calcLayerIdReduced(layerId_);

    // Note module ring in endcap
    endcapRing_ = barrel_ ? 0 : trackerTopology->tidRing(detId);
    if (not barrel_) {
      // Apply bodge, since Topology class annoyingly starts ring count at 1, even in endcap wheels where
      // inner rings are absent.
      unsigned int iWheel = trackerTopology->tidWheel(detId);
      if (iWheel >= 3 && iWheel <= 5)
        endcapRing_ += 3;
    }

    // Note if tilted barrel module & get tilt angle (in range 0 to PI).
    tiltedBarrel_ = barrel_ && (trackerTopology->tobSide(detId) != BarrelModuleType::flat);
    float deltaR = std::abs(R1 - R0);
    float deltaZ = (R1 - R0 > 0) ? (Z1 - Z0) : -(Z1 - Z0);
    tiltAngle_ = atan(deltaR / deltaZ);

    // Get sensor strip or pixel pitch using innermost sensor of pair.

    const Bounds& bounds = det0->surface().bounds();
    sensorWidth_ = bounds.width();  // Width of sensitive region of sensor (= stripPitch * nStrips).
    sensorSpacing_ = sqrt((moduleMaxR_ - moduleMinR_) * (moduleMaxR_ - moduleMinR_) +
                          (moduleMaxZ_ - moduleMinZ_) * (moduleMaxZ_ - moduleMinZ_));
    nStrips_ = specTopol_->nrows();  // No. of strips in sensor
    std::pair<float, float> pitch = specTopol_->pitch();
    stripPitch_ = pitch.first;    // Strip pitch (or pixel pitch along shortest axis)
    stripLength_ = pitch.second;  //  Strip length (or pixel pitch along longest axis)

    // Get module type ID defined by firmware.

    moduleTypeID_ = TrackerModule::calcModuleType(stripPitch_, sensorSpacing_, barrel_, tiltedBarrel_, psModule_);
  }

  //=== Calculate reduced layer ID (in range 1-7), for  packing into 3 bits to simplify the firmware.

  unsigned int TrackerModule::calcLayerIdReduced(unsigned int layerId) {
    // Don't bother distinguishing two endcaps, as no track can have stubs in both.
    unsigned int lay = (layerId < 20) ? layerId : layerId - 10;

    // No genuine track can have stubs in both barrel layer 6 and endcap disk 11 etc., so merge their layer IDs.
    if (lay == 6)
      lay = 11;
    if (lay == 5)
      lay = 12;
    if (lay == 4)
      lay = 13;
    if (lay == 3)
      lay = 15;
    // At this point, the reduced layer ID can have values of 1, 2, 11, 12, 13, 14, 15. So correct to put in range 1-7.
    if (lay > 10)
      lay -= 8;

    if (lay < 1 || lay > 7)
      throw cms::Exception("LogicError") << "TrackerModule: Reduced layer ID out of expected range";

    return lay;
  }

  //=== Get module type ID defined by firmware.

  unsigned int TrackerModule::calcModuleType(
      float pitch, float space, bool barrel, bool tiltedBarrel, bool psModule) const {
    // Calculate unique module type ID, allowing sensor pitch/seperation of module to be determined in FW.

    unsigned int moduleType = 999;
    constexpr float tol = 0.001;  // Tolerance

    for (unsigned int i = 0; i < moduleTypeCfg_.pitchVsType.size(); i++) {
      if (std::abs(pitch - moduleTypeCfg_.pitchVsType[i]) < tol &&
          std::abs(space - moduleTypeCfg_.spaceVsType[i]) < tol && barrel == moduleTypeCfg_.barrelVsType[i] &&
          tiltedBarrel == moduleTypeCfg_.tiltedVsType[i] && psModule == moduleTypeCfg_.psVsType[i]) {
        moduleType = i;
      }
    }

    if (moduleType == 999) {
      std::stringstream text;
      text << "WARNING: TrackerModule found tracker module type unknown to firmware: pitch=" << pitch
           << " separation=" << space << " barrel=" << barrel << " tilted=" << tiltedBarrel << " PS=" << psModule;
      std::call_once(
          printOnce, [](string t) { edm::LogWarning("L1track") << t; }, text.str());
    }
    return moduleType;
  }
}  // namespace tmtt
