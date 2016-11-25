// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBackPlaneCorrectioNFakeESSource
//
/**\class SiStripBackPlaneCorrectioNFakeESSource SiStripBackPlaneCorrectioNFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBackPlaneCorrectioNFakeESSource.cc

 Description: "fake" SiStripBackPlaneCorrection ESProducer - fixed value from configuration for each module geometry

 Implementation:
     Port of SiStripBackPlaneCorrectionGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripBackPlaneCorrectioNFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBackPlaneCorrectioNFakeESSource(const edm::ParameterSet&);
  ~SiStripBackPlaneCorrectioNFakeESSource();

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity );

  typedef std::shared_ptr<SiStripBackPlaneCorrection> ReturnType;
  ReturnType produce(const SiStripBackPlaneCorrectionRcd&);

private:
  std::vector<double> m_valuePerModuleGeometry;
  edm::FileInPath m_file;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

SiStripBackPlaneCorrectioNFakeESSource::SiStripBackPlaneCorrectioNFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBackPlaneCorrectionRcd>();

  m_valuePerModuleGeometry = iConfig.getParameter<std::vector<double>>("BackPlaneCorrection_PerModuleGeometry");
  m_file = iConfig.getParameter<edm::FileInPath>("file");
}

SiStripBackPlaneCorrectioNFakeESSource::~SiStripBackPlaneCorrectioNFakeESSource() {}

void SiStripBackPlaneCorrectioNFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripBackPlaneCorrectioNFakeESSource::ReturnType
SiStripBackPlaneCorrectioNFakeESSource::produce(const SiStripBackPlaneCorrectionRcd& iRecord)
{
  using namespace edm::es;

  edm::ESHandle<TrackerTopology> tTopo;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopo);

  std::shared_ptr<SiStripBackPlaneCorrection> backPlaneCorrection{new SiStripBackPlaneCorrection};

  SiStripDetInfoFileReader reader{m_file.fullPath()};

  for ( const auto& detId : reader.getAllDetIds() ) {
    unsigned int moduleGeometry = tTopo->moduleGeometry(DetId(detId))-1;
    if ( moduleGeometry > m_valuePerModuleGeometry.size() ) {
      edm::LogError("SiStripBackPlaneCorrectionGenerator") << " BackPlaneCorrection_PerModuleGeometry only contains " << m_valuePerModuleGeometry.size() << "elements and module is out of range";
    }
    float value = m_valuePerModuleGeometry[moduleGeometry];
    if ( ! backPlaneCorrection->putBackPlaneCorrection(detId, value) ) {
      edm::LogError("SiStripBackPlaneCorrectionGenerator")<<" detid already exists";
    }
  }

  return backPlaneCorrection;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBackPlaneCorrectioNFakeESSource);
