#include "L1Trigger/TrackerDTC/interface/TTDTCConverter.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>

using namespace std;
using namespace edm;
using namespace TrackerDTC;

TTDTCConverter::TTDTCConverter(const Run& iRun,
                               const EventSetup& iSetup,
                               const string& processName,
                               const string& productLabel) {
  // get iConfig of used TrackerDTC
  const ParameterSet* iConfig = nullptr;
  const pset::Registry* psetRegistry = pset::Registry::instance();
  for (const ProcessConfiguration& pc : iRun.processHistory()) {
    if (!processName.empty() && processName != pc.processName())
      continue;

    const ParameterSet* processPset = psetRegistry->getMapped(pc.parameterSetID());
    if (processPset && processPset->exists(productLabel))
      iConfig = &processPset->getParameterSet(productLabel);
  }

  if (!iConfig) {
    cms::Exception exception("Configuration", "TrackerDTC config not found in process history.");
    exception.addContext("L1TrackerDTC::Settings::beginRun");
    exception.addAdditionalInfo("Used process name: " + processName + ", used product label: " + productLabel + ".");

    throw exception;
  }

  settings_ = make_unique<Settings>(*iConfig);

  settings_->beginRun(iRun, iSetup);
}

GlobalPoint TTDTCConverter::pos(const TTDTC::Frame& frame, const int& region) const {
  GlobalPoint p;
  if (frame.first.isNull())
    return p;

  TTBV bv(frame.second);

  if (settings_->dataFormat() == "Hybrid") {
    const TrackerGeometry* trackerGeometry = settings_->trackerGeometry();
    const TrackerTopology* trackerTopology = settings_->trackerTopology();
    SettingsHybrid* format = settings_->hybrid();
    const vector<double>& diskZs = format->diskZs();
    const vector<double>& layerRs = format->layerRs();

    const DetId detId = frame.first->getDetId();
    const bool barrel = detId.subdetId() == StripSubdetector::TOB;
    const bool psModule = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    const int layerId = barrel ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + 10;

    SettingsHybrid::SensorType type;
    if (barrel && psModule)
      type = SettingsHybrid::barrelPS;
    if (barrel && !psModule)
      type = SettingsHybrid::barrel2S;
    if (!barrel && psModule)
      type = SettingsHybrid::diskPS;
    if (!barrel && !psModule)
      type = SettingsHybrid::disk2S;

    bv >>= 1 + settings_->widthLayer() + format->widthBend(type) + format->widthAlpha(type);

    double phi = (bv.val(format->widthPhi(type), 0, true) + .5) * format->basePhi(type);
    bv >>= format->widthPhi(type);
    double z = (bv.val(format->widthZ(type), 0, true) + .5) * format->baseZ(type);
    bv >>= format->widthZ(type);
    double r = (bv.val(format->widthR(type), 0, true) + .5) * format->baseR(type);

    r += barrel ? layerRs.at(layerId - 1) : 0.;
    z += barrel ? 0. : diskZs.at(layerId - 11);

    phi = reco::deltaPhi(phi + region * settings_->baseRegion(), 0.);

    if (type == SettingsHybrid::disk2S) {
      r = bv.val(format->widthR(type));
      r = format->disk2SR(layerId - 11, (int)r);
    }

    p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  }

  if (settings_->dataFormat() == "TMTT") {
    SettingsTMTT* format_ = settings_->tmtt();

    bv >>=
        2 * format_->widthQoverPtBin() + 2 * settings_->widthEta() + format_->numSectorsPhi() + settings_->widthLayer();

    double z = (bv.val(settings_->widthZ(), 0, true) + .5) * settings_->baseZ();
    bv >>= settings_->widthZ();
    double phi = (bv.val(settings_->widthPhi(), 0, true) + .5) * settings_->basePhi();
    bv >>= settings_->widthPhi();
    double r = (bv.val(settings_->widthR(), 0, true) + .5) * settings_->baseR();
    bv >>= settings_->widthR();

    r = r + settings_->chosenRofPhi();
    phi = reco::deltaPhi(phi + region * settings_->baseRegion(), 0.);

    p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  }

  return p;
}