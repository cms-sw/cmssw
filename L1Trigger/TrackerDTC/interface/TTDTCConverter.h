#ifndef L1Trigger_TrackerDTC_TTDTCConverter_h
#define L1Trigger_TrackerDTC_TTDTCConverter_h

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"

#include <vector>
#include <string>

// returns bit accurate position of a stub from a given processing region [0-8] (phi slice of outer tracker)
GlobalPoint TTDTCConverter(trackerDTC::Settings* settings, const TTDTC::Frame& frame, int region) {
  GlobalPoint p;
  if (frame.first.isNull())
    return p;
  TTBV bv(frame.second);

  if (settings->dataFormat() == "Hybrid") {
    const TrackerGeometry* trackerGeometry = settings->trackerGeometry();
    const TrackerTopology* trackerTopology = settings->trackerTopology();
    trackerDTC::SettingsHybrid* format = settings->hybrid();
    const std::vector<double>& diskZs = format->diskZs();
    const std::vector<double>& layerRs = format->layerRs();

    const DetId detId(frame.first->getDetId() + settings->offsetDetIdDSV());
    const bool barrel = detId.subdetId() == StripSubdetector::TOB;
    const bool psModule = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    const int layerId = barrel ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + 10;
    const GeomDetUnit* det = trackerGeometry->idToDetUnit(detId);
    const bool side = det->position().z() >= 0.;

    trackerDTC::SettingsHybrid::SensorType type;
    if (barrel && psModule)
      type = trackerDTC::SettingsHybrid::barrelPS;
    if (barrel && !psModule)
      type = trackerDTC::SettingsHybrid::barrel2S;
    if (!barrel && psModule)
      type = trackerDTC::SettingsHybrid::diskPS;
    if (!barrel && !psModule)
      type = trackerDTC::SettingsHybrid::disk2S;

    bv >>= 1 + settings->widthLayer() + format->widthBend(type) + format->widthAlpha(type);

    double phi = (bv.val(format->widthPhi(type), 0, true) + .5) * format->basePhi(type);
    bv >>= format->widthPhi(type);
    double z = (bv.val(format->widthZ(type), 0, true) + .5) * format->baseZ(type);
    bv >>= format->widthZ(type);
    double r = (bv.val(format->widthR(type), 0, barrel) + .5) * format->baseR(type);

    if (barrel) {
      r += layerRs.at(layerId - settings->offsetLayerId());
    } else {
      z += diskZs.at(layerId - settings->offsetLayerId() - settings->offsetLayerDisks()) * (side ? 1. : -1.);
    }

    phi = reco::deltaPhi(phi + region * settings->baseRegion(), 0.);

    if (type == trackerDTC::SettingsHybrid::disk2S) {
      r = bv.val(format->widthR(type));
      r = format->disk2SR(layerId - settings->offsetLayerId() - settings->offsetLayerDisks(), (int)r);
    }

    p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  } else if (settings->dataFormat() == "TMTT") {
    trackerDTC::SettingsTMTT* format_ = settings->tmtt();

    bv >>=
        2 * format_->widthQoverPtBin() + 2 * settings->widthEta() + format_->numSectorsPhi() + settings->widthLayer();

    double z = (bv.val(settings->widthZ(), 0, true) + .5) * settings->baseZ();
    bv >>= settings->widthZ();
    double phi = (bv.val(settings->widthPhi(), 0, true) + .5) * settings->basePhi();
    bv >>= settings->widthPhi();
    double r = (bv.val(settings->widthR(), 0, true) + .5) * settings->baseR();
    bv >>= settings->widthR();

    r = r + settings->chosenRofPhi();
    phi = reco::deltaPhi(phi + region * settings->baseRegion(), 0.);

    p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  }

  return p;
}

#endif