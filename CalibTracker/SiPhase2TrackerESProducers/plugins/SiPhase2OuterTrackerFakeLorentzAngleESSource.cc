// -*- C++ -*-
//
// Package:    SiPhase2OuterTrackerFakeLorentzAngleESSource
// Class:      SiPhase2OuterTrackerFakeLorentzAngleESSource
//
/**\class SiPhase2OuterTrackerFakeLorentzAngleESSource SiPhase2OuterTrackerFakeLorentzAngleESSource.h CalibTracker/SiPhase2TrackerESProducers/src/SiPhase2OuterTrackerFakeLorentzAngleESSource.cc
 Description: <one line class summary>
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco Musich
//         Created:  Jul 31st, 2020
//
//

// user include files
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CalibTracker/SiPhase2TrackerESProducers/interface/SiPhase2OuterTrackerFakeLorentzAngleESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"
//
// constructors and destructor
//
SiPhase2OuterTrackerFakeLorentzAngleESSource::SiPhase2OuterTrackerFakeLorentzAngleESSource(
    const edm::ParameterSet& conf_)
    : LAvalue_(conf_.getParameter<double>("LAValue")) {
  edm::LogInfo("SiPhase2OuterTrackerFakeLorentzAngleESSource::SiPhase2OuterTrackerFakeLorentzAngleESSource");
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this).setConsumes(m_tTopoToken).setConsumes(m_geomDetToken);
  findingRecord<SiPhase2OuterTrackerLorentzAngleRcd>();
}

SiPhase2OuterTrackerFakeLorentzAngleESSource::~SiPhase2OuterTrackerFakeLorentzAngleESSource() {}

std::unique_ptr<SiPhase2OuterTrackerLorentzAngle> SiPhase2OuterTrackerFakeLorentzAngleESSource::produce(
    const SiPhase2OuterTrackerLorentzAngleRcd& iRecord) {
  using namespace edm::es;
  SiPhase2OuterTrackerLorentzAngle* obj = new SiPhase2OuterTrackerLorentzAngle();

  const auto& geomDet = iRecord.getRecord<TrackerTopologyRcd>().get(m_geomDetToken);
  for (const auto detId : TrackerGeometryUtils::getSiStripDetIds(geomDet)) {
    const DetId detectorId = DetId(detId);
    const int subDet = detectorId.subdetId();
    if (detectorId.det() == DetId::Detector::Tracker) {
      if (subDet == StripSubdetector::TOB || subDet == StripSubdetector::TID) {
        if (!obj->putLorentzAngle(detId, LAvalue_))
          edm::LogError("SiPhase2OuterTrackerFakeLorentzAngleESSource")
              << "[SiPhase2OuterTrackerFakeLorentzAngleESSource::produce] detid already exists" << std::endl;

      }  // if it's a OT DetId
    }    // check if Tracker
  }      // loop on DetIds

  return std::unique_ptr<SiPhase2OuterTrackerLorentzAngle>(obj);
}

void SiPhase2OuterTrackerFakeLorentzAngleESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                                  const edm::IOVSyncValue& iosv,
                                                                  edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPhase2OuterTrackerFakeLorentzAngleESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("LAValue", 0.07);
  descriptions.add("siPhase2OTFakeLorentzAngleESSource", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiPhase2OuterTrackerFakeLorentzAngleESSource);
