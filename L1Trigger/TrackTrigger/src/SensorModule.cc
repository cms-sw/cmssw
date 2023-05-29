#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <cmath>
#include <algorithm>
#include <iterator>
#include <vector>

using namespace std;
using namespace edm;

namespace tt {

  SensorModule::SensorModule(const Setup* setup, const DetId& detId, int dtcId, int modId)
      : detId_(detId), dtcId_(dtcId), modId_(modId) {
    const TrackerGeometry* trackerGeometry = setup->trackerGeometry();
    const TrackerTopology* trackerTopology = setup->trackerTopology();
    const GeomDetUnit* geomDetUnit = trackerGeometry->idToDetUnit(detId);
    const PixelTopology* pixelTopology =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit)->surface();
    const GlobalPoint pos0 = GlobalPoint(geomDetUnit->position());
    const GlobalPoint pos1 =
        GlobalPoint(trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId))->position());
    // detector region [0-8]
    const int region = dtcId_ / setup->numDTCsPerRegion();
    // module radius in cm
    r_ = pos0.perp();
    // module phi w.r.t. detector region_ centre in rad
    phi_ = deltaPhi(pos0.phi() - (region + .5) * setup->baseRegion());
    // module z in cm
    z_ = pos0.z();
    // sensor separation in cm
    sep_ = (pos1 - pos0).mag();
    // sensor pitch in cm [strip=.009,pixel=.01]
    pitchRow_ = pixelTopology->pitch().first;
    // sensor length in cm [strip=5,pixel=.15625]
    pitchCol_ = pixelTopology->pitch().second;
    // number of columns [2S=2,PS=8]
    numColumns_ = pixelTopology->ncolumns();
    // number of rows [2S=8*127,PS=8*120]
    numRows_ = pixelTopology->nrows();
    // +z or -z
    side_ = pos0.z() >= 0.;
    // main sensor inside or outside
    flipped_ = pos0.mag() > pos1.mag();
    // barrel or endcap
    barrel_ = detId.subdetId() == StripSubdetector::TOB;
    // Pixel-Strip or 2Strip module
    psModule_ = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    // module tilt angle measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    tilt_ = flipped_ ? atan2(pos1.z() - pos0.z(), pos0.perp() - pos1.perp())
                     : atan2(pos0.z() - pos1.z(), pos1.perp() - pos0.perp());
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    sinTilt_ = std::sin(tilt_);
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    cosTilt_ = std::cos(tilt_);
    // layer id [barrel: 0-5, endcap: 0-4]
    const int layer =
        (barrel_ ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId)) - setup->offsetLayerId();
    // layer id [1-6,11-15]
    layerId_ = layer + setup->offsetLayerId() + (barrel_ ? 0 : setup->offsetLayerDisks());
    // TTStub row needs flip of sign
    signRow_ = signbit(deltaPhi(plane.rotation().x().phi() - pos0.phi()));
    // TTStub col needs flip of sign
    signCol_ = !barrel_ && !side_;
    // TTStub bend needs flip of sign
    signBend_ = barrel_ || (!barrel_ && side_);
    // determing sensor type
    if (barrel_ && psModule_)
      type_ = BarrelPS;
    if (barrel_ && !psModule_)
      type_ = Barrel2S;
    if (!barrel_ && psModule_)
      type_ = DiskPS;
    if (!barrel_ && !psModule_)
      type_ = Disk2S;
    // encoding for 2S endcap radii
    encodedR_ = -1;
    if (type_ == Disk2S) {
      const int offset = setup->hybridNumRingsPS(layer);
      const int ring = trackerTopology->tidRing(detId);
      encodedR_ = numColumns_ * (ring - offset);
    }
    // r and z offsets
    if (barrel_) {
      offsetR_ = setup->hybridLayerR(layer);
      offsetZ_ = 0.;
    } else {
      offsetR_ = 0.;
      offsetZ_ = side_ ? setup->hybridDiskZ(layer) : -setup->hybridDiskZ(layer);
    }
    const TypeTilt typeTilt = static_cast<TypeTilt>(trackerTopology->tobSide(detId));
    // getting bend window size
    double windowSize(-1.);
    if (barrel_) {
      if (typeTilt == flat)
        windowSize = setup->windowSizeBarrelLayer(layerId_);
      else {
        int ladder = trackerTopology->tobRod(detId);
        if (typeTilt == tiltedMinus)
          // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
          ladder = 1 + setup->numTiltedLayerRing(layerId_) - ladder;
        windowSize = setup->windowSizeTiltedLayerRing(layerId_, ladder);
      }
    } else {
      const int ring = trackerTopology->tidRing(detId);
      const int lay = layer + setup->offsetLayerId();
      windowSize = setup->windowSizeEndcapDisksRing(lay, ring);
    }
    windowSize_ = windowSize / setup->baseWindowSize();
    // calculate tilt correction parameter used to project r to z uncertainty
    tiltCorrectionSlope_ = barrel_ ? 0. : 1.;
    tiltCorrectionIntercept_ = barrel_ ? 1. : 0.;
    if (typeTilt == tiltedMinus || typeTilt == tiltedPlus) {
      tiltCorrectionSlope_ = setup->tiltApproxSlope();
      tiltCorrectionIntercept_ = setup->tiltApproxIntercept();
    }
  }

}  // namespace tt