#include "L1Trigger/TrackerDTC/interface/Module.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <cmath>

using namespace std;
using namespace edm;

namespace TrackerDTC {

  Module::Module(Settings* settings, const DetId& detId, const int& modId)
      :  // outer tracker dtc routing block id [0-1]
        blockId_((modId % settings->numModulesPerDTC()) / settings->numModulesPerRoutingBlock()),
        // routing block channel id [0-35]
        channelId_((modId % settings->numModulesPerDTC()) % settings->numModulesPerRoutingBlock()) {
    const int region = modId / (settings->numModulesPerDTC() * settings->numDTCsPerRegion());  // detector region [0-8]

    const TrackerGeometry* trackerGeometry = settings->trackerGeometry();
    const TrackerTopology* trackerTopology = settings->trackerTopology();

    const GeomDetUnit* det0 = trackerGeometry->idToDetUnit(detId);
    const GlobalPoint pos0 = GlobalPoint(det0->position());
    const GlobalPoint pos1 =
        GlobalPoint(trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId))->position());
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det0)->specificTopology()));
    const Plane plane = dynamic_cast<const PixelGeomDetUnit*>(det0)->surface();

    R_ = pos0.perp();  // module radius in cm
    // module phi w.r.t. detector region centre in rad
    Phi_ = deltaPhi(pos0.phi() - (region + .5) * settings->baseRegion());
    Z_ = pos0.z();                                        // module z in cm
    sep_ = (pos1 - pos0).mag();                           // sensor separation in cm
    pitchRow_ = topol->pitch().first;                     // sensor pitch in cm [strip=.009,pixel=.01]
    pitchCol_ = topol->pitch().second;                    // sensor length in cm [strip=5,pixel=.15625]
    numColumns_ = topol->ncolumns();                      // number of columns [2S=2,PS=8]
    numRows_ = topol->nrows();                            // number of rows [2S=8*127,PS=8*120]
    side_ = pos0.z() > 0.;                                // +z or -z
    flipped_ = pos0.mag() > pos1.mag();                   // main sensor inside or outside
    barrel_ = detId.subdetId() == StripSubdetector::TOB;  // barrel or endcap
    // module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    tilt_ = flipped_ ? atan2(pos1.z() - pos0.z(), pos0.perp() - pos1.perp())
                     : atan2(pos0.z() - pos1.z(), pos1.perp() - pos0.perp());
    sin_ = sin(tilt_);
    cos_ = cos(tilt_);
    layerId_ = barrel_ ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + 10;  // layer id [1-6,11-15]

    signRow_ = signbit(deltaPhi(plane.rotation().x().phi() - pos0.phi()));  // TTStub row needs flip of sign
    signCol_ = !barrel_ && !side_;                                          // TTStub col needs flip of sign
    signBend_ = barrel_ || (!barrel_ && side_);                             // TTStub bend needs flip of sign

    if (settings->dataFormat() != "Hybrid")
      return;

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
    const TypeBarrel type = static_cast< TypeBarrel >( trackerTopology->tobSide(detId) );

    int ladder = barrel_ ? trackerTopology->tobRod(detId) : trackerTopology->tidRing(detId);
    if (barrel_ && type == tiltedMinus)
      // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
      ladder = 1 + numTiltedLayerRings.at(layerId_) - ladder;

    double windowSize =
        barrel_ ? windowSizeBarrelLayers.at(layerId_) : windowSizeEndcapDisksRings.at(layerId_ - settings->offsetLayerDisks()).at(ladder);
    if (barrel_ && type != flat)
      windowSize = windowSizeTiltedLayerRings.at(layerId_).at(ladder);
    const int ws = windowSize / format->baseWindowSize();

    // getting bend encoding
    // index = encoded bend, value = decoded bend
    bendEncoding_ = psModule ? bendEncodingsPS.at(ws) : bendEncodings2S.at(ws);

    // encoding for 2S endcap radii
    decodedR_ = type_ == SettingsHybrid::disk2S ? numColumns_ * (ladder - numRingsPS.at(layerId_ - settings->offsetLayerId() - settings->offsetLayerDisks())) : 0;

    // r and z offsets
    offsetR_ = barrel_ ? layerRs.at(layerId_ - settings->offsetLayerId()) : 0.;
    offsetZ_ = barrel_ ? 0. : diskZs.at(layerId_ - settings->offsetLayerId() - settings->offsetLayerDisks());

    // layer id encoding
    const int dtcId = modId / settings->numModulesPerDTC();
    // index = decoded layerId, value = encoded layerId
    const vector<int>& layerIdEncoding = layerIdEncodings.at(dtcId);
    layerId_ = distance(layerIdEncoding.begin(), find(layerIdEncoding.begin(), layerIdEncoding.end(), layerId_));
  }

}  // namespace TrackerDTC
