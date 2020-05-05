#include "L1Trigger/TrackerDTC/interface/Module.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <cmath>
#include <vector>

using namespace std;
using namespace edm;

namespace trackerDTC {

  Module::Module(Settings* settings, const DetId& detId, int dtcId) {
    const TrackerGeometry* trackerGeometry = settings->trackerGeometry();
    const TrackerTopology* trackerTopology = settings->trackerTopology();
    const GeomDetUnit* det0 = trackerGeometry->idToDetUnit(detId);
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det0)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det0)->surface();
    const GlobalPoint pos0 = GlobalPoint(det0->position());
    const GlobalPoint pos1 =
        GlobalPoint(trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId))->position());
    // detector region_ [0-8]
    region_ = dtcId / settings->numDTCsPerRegion();
    // module radius in cm
    R_ = pos0.perp();
    // module phi w.r.t. detector region_ centre in rad
    Phi_ = deltaPhi(pos0.phi() - (region_ + .5) * settings->baseRegion());
    // module z in cm
    Z_ = pos0.z();
    // sensor separation in cm
    sep_ = (pos1 - pos0).mag();
    // sensor pitch in cm [strip=.009,pixel=.01]
    pitchRow_ = topol->pitch().first;
    // sensor length in cm [strip=5,pixel=.15625]
    pitchCol_ = topol->pitch().second;
    // number of columns [2S=2,PS=8]
    numColumns_ = topol->ncolumns();
    // number of rows [2S=8*127,PS=8*120]
    numRows_ = topol->nrows();
    side_ = pos0.z() >= 0.;
    flipped_ = pos0.mag() > pos1.mag();
    barrel_ = detId.subdetId() == StripSubdetector::TOB;
    // module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    tilt_ = flipped_ ? atan2(pos1.z() - pos0.z(), pos0.perp() - pos1.perp())
                     : atan2(pos0.z() - pos1.z(), pos1.perp() - pos0.perp());
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    sin_ = sin(tilt_);
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    cos_ = cos(tilt_);
    // layer id [1-6,11-15]
    layerId_ =
        barrel_ ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + settings->offsetLayerDisks();
    // TTStub row needs flip of sign
    signRow_ = signbit(deltaPhi(plane.rotation().x().phi() - pos0.phi()));
    // TTStub col needs flip of sign
    signCol_ = !barrel_ && !side_;
    // TTStub bend needs flip of sign
    signBend_ = barrel_ || (!barrel_ && side_);

    // sets hybrid specific member
    if (settings->dataFormat() != "Hybrid")
      return;
    // gettings hybrid config
    SettingsHybrid* format = settings->hybrid();
    const vector<int>& numRingsPS = format->numRingsPS();
    const vector<double>& diskZs = format->diskZs();
    const vector<double>& layerRs = format->layerRs();
    const vector<double>& numTiltedLayerRings = format->numTiltedLayerRings();
    const vector<double>& windowSizeBarrelLayers = format->windowSizeBarrelLayers();
    const vector<vector<double> >& windowSizeTiltedLayerRings = format->windowSizeTiltedLayerRings();
    const vector<vector<double> >& windowSizeEndcapDisksRings = format->windowSizeEndcapDisksRings();
    const vector<vector<double> >& bendEncodingsPS = format->bendEncodingsPS();
    const vector<vector<double> >& bendEncodings2S = format->bendEncodings2S();
    const vector<vector<int> >& layerIdEncodings = format->layerIdEncodings();
    // determing sensor type
    const bool psModule = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    if (barrel_ && psModule)
      type_ = SettingsHybrid::barrelPS;
    if (barrel_ && !psModule)
      type_ = SettingsHybrid::barrel2S;
    if (!barrel_ && psModule)
      type_ = SettingsHybrid::diskPS;
    if (!barrel_ && !psModule)
      type_ = SettingsHybrid::disk2S;
    // getting windows size for bend encoding
    enum TypeBarrel { nonBarrel = 0, tiltedMinus = 1, tiltedPlus = 2, flat = 3 };
    const TypeBarrel type = static_cast<TypeBarrel>(trackerTopology->tobSide(detId));
    int ladder = barrel_ ? trackerTopology->tobRod(detId) : trackerTopology->tidRing(detId);
    if (barrel_ && type == tiltedMinus)
      // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
      ladder = 1 + numTiltedLayerRings.at(layerId_) - ladder;
    double windowSize = barrel_ ? windowSizeBarrelLayers.at(layerId_)
                                : windowSizeEndcapDisksRings.at(layerId_ - settings->offsetLayerDisks()).at(ladder);
    if (barrel_ && type != flat)
      windowSize = windowSizeTiltedLayerRings.at(layerId_).at(ladder);
    const int ws = windowSize / format->baseWindowSize();
    // getting bend encoding
    bendEncoding_ = psModule ? bendEncodingsPS.at(ws) : bendEncodings2S.at(ws);
    // encoding for 2S endcap radii
    decodedR_ = -1;
    if (type_ == SettingsHybrid::disk2S) {
      const int offset = numRingsPS.at(layerId_ - settings->offsetLayerId() - settings->offsetLayerDisks());
      decodedR_ = numColumns_ * (ladder - offset);
    }
    // r and z offsets
    offsetR_ = barrel_ ? layerRs.at(layerId_ - settings->offsetLayerId()) : 0.;
    offsetZ_ = barrel_ ? 0. : diskZs.at(layerId_ - settings->offsetLayerId() - settings->offsetLayerDisks());
    if (!side_)
      offsetZ_ *= -1.;
    // layer id encoding
    const vector<int>& layerIdEncoding = layerIdEncodings.at(dtcId % settings->numDTCsPerRegion());
    layerId_ = distance(layerIdEncoding.begin(), find(layerIdEncoding.begin(), layerIdEncoding.end(), layerId_));
  }

}  // namespace trackerDTC