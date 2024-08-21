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
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CalibTracker/SiPhase2TrackerESProducers/interface/SiPhase2OuterTrackerFakeLorentzAngleESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"
//
// constructors and destructor
//
SiPhase2OuterTrackerFakeLorentzAngleESSource::SiPhase2OuterTrackerFakeLorentzAngleESSource(
    const edm::ParameterSet& conf_)
    : LAvalue_(conf_.getParameter<double>("LAValue")), recordName_(conf_.getParameter<std::string>("recordName")) {
  edm::LogInfo("SiPhase2OuterTrackerFakeLorentzAngleESSource::SiPhase2OuterTrackerFakeLorentzAngleESSource");
  // the following line is needed to tell the framework what
  // data is being produced
  if (recordName_ == "LorentzAngle") {
    auto cc = setWhatProduced(this, &SiPhase2OuterTrackerFakeLorentzAngleESSource::produceOTLA);
    m_tTopoToken = cc.consumes();
    m_geomDetToken = cc.consumes();
    findingRecord<SiPhase2OuterTrackerLorentzAngleRcd>();
  } else if (recordName_ == "SimLorentzAngle") {
    auto cc = setWhatProduced(this, &SiPhase2OuterTrackerFakeLorentzAngleESSource::produceOTSimLA);
    m_tTopoToken = cc.consumes();
    m_geomDetToken = cc.consumes();
    findingRecord<SiPhase2OuterTrackerLorentzAngleSimRcd>();
  }
}

SiPhase2OuterTrackerFakeLorentzAngleESSource::~SiPhase2OuterTrackerFakeLorentzAngleESSource() {}

namespace fakeOTLA {
  template <class T>
  std::unique_ptr<T> produceRecord(const float value, const GeometricDet& geomDet) {
    using namespace edm::es;
    T* obj = new T();
    for (const auto detId : TrackerGeometryUtils::getOuterTrackerDetIds(geomDet)) {
      const DetId detectorId = DetId(detId);
      const int subDet = detectorId.subdetId();
      if (detectorId.det() == DetId::Detector::Tracker) {
        if (subDet == StripSubdetector::TOB || subDet == StripSubdetector::TID) {
          if (!obj->putLorentzAngle(detId, value))
            edm::LogError("SiPhase2OuterTrackerFakeLorentzAngleESSource")
                << "[SiPhase2OuterTrackerFakeLorentzAngleESSource::produce] detid already exists" << std::endl;
        }  // if it's a OT DetId
      }    // check if Tracker
    }      // loop on DetIds
    return std::unique_ptr<T>(obj);
  }
}  // namespace fakeOTLA

std::unique_ptr<SiPhase2OuterTrackerLorentzAngle> SiPhase2OuterTrackerFakeLorentzAngleESSource::produceOTLA(
    const SiPhase2OuterTrackerLorentzAngleRcd& rcd) {
  const auto& geomDet = rcd.getRecord<TrackerTopologyRcd>().get(m_geomDetToken);
  return fakeOTLA::produceRecord<SiPhase2OuterTrackerLorentzAngle>(LAvalue_, geomDet);
}

std::unique_ptr<SiPhase2OuterTrackerLorentzAngle> SiPhase2OuterTrackerFakeLorentzAngleESSource::produceOTSimLA(
    const SiPhase2OuterTrackerLorentzAngleSimRcd& rcd) {
  const auto& geomDet = rcd.getRecord<TrackerTopologyRcd>().get(m_geomDetToken);
  return fakeOTLA::produceRecord<SiPhase2OuterTrackerLorentzAngle>(LAvalue_, geomDet);
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
  desc.add<std::string>("recordName", "LorentzAngle");
  descriptions.add("siPhase2OTFakeLorentzAngleESSource", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiPhase2OuterTrackerFakeLorentzAngleESSource);
