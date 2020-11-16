// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBackPlaneCorrectionFakeESSource
//
/**\class SiStripBackPlaneCorrectionFakeESSource SiStripBackPlaneCorrectionFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBackPlaneCorrectionFakeESSource.cc

 Description: "fake" SiStripBackPlaneCorrection ESProducer - fixed value from configuration for each module geometry

 Implementation:
     Port of SiStripBackPlaneCorrectionGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"

class SiStripBackPlaneCorrectionFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBackPlaneCorrectionFakeESSource(const edm::ParameterSet&);
  ~SiStripBackPlaneCorrectionFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripBackPlaneCorrection> ReturnType;
  ReturnType produce(const SiStripBackPlaneCorrectionRcd&);

private:
  std::vector<double> m_valuePerModuleGeometry;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> m_geomDetToken;
};

SiStripBackPlaneCorrectionFakeESSource::SiStripBackPlaneCorrectionFakeESSource(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);
  m_tTopoToken = cc.consumes();
  m_geomDetToken = cc.consumes();
  findingRecord<SiStripBackPlaneCorrectionRcd>();

  m_valuePerModuleGeometry = iConfig.getParameter<std::vector<double>>("BackPlaneCorrection_PerModuleGeometry");
}

SiStripBackPlaneCorrectionFakeESSource::~SiStripBackPlaneCorrectionFakeESSource() {}

void SiStripBackPlaneCorrectionFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                            const edm::IOVSyncValue& iov,
                                                            edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripBackPlaneCorrectionFakeESSource::ReturnType SiStripBackPlaneCorrectionFakeESSource::produce(
    const SiStripBackPlaneCorrectionRcd& iRecord) {
  using namespace edm::es;

  const auto& geomDet = iRecord.getRecord<TrackerTopologyRcd>().get(m_geomDetToken);
  const auto& tTopo = iRecord.get(m_tTopoToken);

  auto backPlaneCorrection = std::make_unique<SiStripBackPlaneCorrection>();

  for (const auto detId : TrackerGeometryUtils::getSiStripDetIds(geomDet)) {
    const auto moduleGeometry = static_cast<unsigned int>(tTopo.moduleGeometry(DetId(detId))) - 1;
    if (moduleGeometry > m_valuePerModuleGeometry.size()) {
      edm::LogError("SiStripBackPlaneCorrectionGenerator")
          << " BackPlaneCorrection_PerModuleGeometry only contains " << m_valuePerModuleGeometry.size()
          << "elements and module is out of range";
    }
    float value = m_valuePerModuleGeometry[moduleGeometry];
    if (!backPlaneCorrection->putBackPlaneCorrection(detId, value)) {
      edm::LogError("SiStripBackPlaneCorrectionGenerator") << " detid already exists";
    }
  }

  return backPlaneCorrection;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBackPlaneCorrectionFakeESSource);
