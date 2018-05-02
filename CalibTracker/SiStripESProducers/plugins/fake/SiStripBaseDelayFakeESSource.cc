// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBaseDelayFakeESSource
//
/**\class SiStripBaseDelayFakeESSource SiStripBaseDelayFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBaseDelayFakeESSource.cc

 Description: "fake" SiStripBaseDelay ESProducer - fixed values from configuration for base delay

 Implementation:
     Port of SiStripBaseDelayGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripBaseDelayFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBaseDelayFakeESSource(const edm::ParameterSet&);
  ~SiStripBaseDelayFakeESSource() override;

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity ) override;

  typedef std::unique_ptr<SiStripBaseDelay> ReturnType;
  ReturnType produce(const SiStripBaseDelayRcd&);

private:
  uint16_t m_coarseDelay;
  uint16_t m_fineDelay;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripBaseDelayFakeESSource::SiStripBaseDelayFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBaseDelayRcd>();

  m_coarseDelay = iConfig.getParameter<uint32_t>("CoarseDelay");
  m_fineDelay = iConfig.getParameter<uint32_t>("FineDelay");
}

SiStripBaseDelayFakeESSource::~SiStripBaseDelayFakeESSource() {}

void SiStripBaseDelayFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripBaseDelayFakeESSource::ReturnType
SiStripBaseDelayFakeESSource::produce(const SiStripBaseDelayRcd& iRecord)
{
  using namespace edm::es;

  auto baseDelay = std::make_unique<SiStripBaseDelay>();

  const edm::Service<SiStripDetInfoFileReader> reader;
  const auto& detInfos = reader->getAllData();
  if ( detInfos.empty() ) {
    edm::LogError("SiStripBaseDelayGenerator") << "Error: detInfo map is empty.";
  }
  for ( const auto& elm : reader->getAllData() ) {
    baseDelay->put(elm.first, m_coarseDelay, m_fineDelay);
  }

  return baseDelay;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBaseDelayFakeESSource);
