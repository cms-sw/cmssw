// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripLatencyFakeESSource
//
/**\class SiStripLatencyFakeESSource SiStripLatencyFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripLatencyFakeESSource.cc

 Description: "fake" SiStripLatency ESProducer - fixed value from configuration for the latency and mode

 Implementation:
     Port of SiStripLatencyGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

class SiStripLatencyFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripLatencyFakeESSource(const edm::ParameterSet&);
  ~SiStripLatencyFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripLatency> ReturnType;
  ReturnType produce(const SiStripLatencyRcd&);

private:
  uint32_t m_latency;
  uint32_t m_mode;
  SiStripDetInfo m_detInfo;
};

SiStripLatencyFakeESSource::SiStripLatencyFakeESSource(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
  findingRecord<SiStripLatencyRcd>();

  m_latency = iConfig.getParameter<uint32_t>("latency");
  m_mode = iConfig.getParameter<uint32_t>("mode");
  m_detInfo = SiStripDetInfoFileReader::read(iConfig.getParameter<edm::FileInPath>("SiStripDetInfoFile").fullPath());
}

SiStripLatencyFakeESSource::~SiStripLatencyFakeESSource() {}

void SiStripLatencyFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                const edm::IOVSyncValue& iov,
                                                edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripLatencyFakeESSource::ReturnType SiStripLatencyFakeESSource::produce(const SiStripLatencyRcd& iRecord) {
  using namespace edm::es;

  auto latency = std::make_unique<SiStripLatency>();

  const auto& detInfos = m_detInfo.getAllData();
  // Take the last detId. Since the map is sorted it will be the biggest value
  if (!detInfos.empty()) {
    // Set the apv number as 6, the highest possible
    edm::LogInfo("SiStripLatencyGenerator") << "detId = " << detInfos.rbegin()->first << " apv = " << 6
                                            << " latency = " << m_latency << " mode = " << m_mode;
    latency->put(detInfos.rbegin()->first, 6, m_latency, m_mode);

    // Call this method to collapse all consecutive detIdAndApvs with the same latency and mode to a single entry
    latency->compress();
  } else {
    edm::LogError("SiStripLatencyGenerator") << "Error: detInfo map is empty. Cannot get the last detId.";
  }

  return latency;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripLatencyFakeESSource);
