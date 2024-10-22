// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripPedestalsFakeESSource
//
/**\class SiStripPedestalsFakeESSource SiStripPedestalsFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripPedestalsFakeESSource.cc

 Description: "fake" SiStripPedestals ESProducer - fixed value from configuration for all pedestals

 Implementation:
     Port of SiStripPedestalsGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

class SiStripPedestalsFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripPedestalsFakeESSource(const edm::ParameterSet&);
  ~SiStripPedestalsFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripPedestals> ReturnType;
  ReturnType produce(const SiStripPedestalsRcd&);

private:
  uint32_t m_pedestalValue;
  uint32_t m_printDebug;
  SiStripDetInfo m_detInfo;
};

SiStripPedestalsFakeESSource::SiStripPedestalsFakeESSource(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
  findingRecord<SiStripPedestalsRcd>();

  m_pedestalValue = iConfig.getParameter<uint32_t>("PedestalValue");
  m_printDebug = iConfig.getUntrackedParameter<uint32_t>("printDebug", 5);
  m_detInfo = SiStripDetInfoFileReader::read(iConfig.getParameter<edm::FileInPath>("SiStripDetInfoFile").fullPath());
}

SiStripPedestalsFakeESSource::~SiStripPedestalsFakeESSource() {}

void SiStripPedestalsFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                  const edm::IOVSyncValue& iov,
                                                  edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripPedestalsFakeESSource::ReturnType SiStripPedestalsFakeESSource::produce(const SiStripPedestalsRcd& iRecord) {
  using namespace edm::es;

  auto pedestals = std::make_unique<SiStripPedestals>();

  uint32_t count{0};
  for (const auto& elm : m_detInfo.getAllData()) {
    //Generate Noises for det detid
    SiStripPedestals::InputVector theSiStripVector;
    for (unsigned short j{0}; j < 128 * elm.second.nApvs; ++j) {
      if (count < m_printDebug) {
        edm::LogInfo("SiStripPedestalsFakeESSource::makePedestals(): ")
            << "detid: " << elm.first << " strip: " << j << " ped: " << m_pedestalValue;
      }
      pedestals->setData(m_pedestalValue, theSiStripVector);
    }
    ++count;
    if (!pedestals->put(elm.first, theSiStripVector)) {
      edm::LogError("SiStripPedestalsFakeESSource::produce ") << " detid already exists";
    }
  }

  return pedestals;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripPedestalsFakeESSource);
