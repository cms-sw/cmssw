// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripThresholdFakeESSource
//
/**\class SiStripThresholdFakeESSource SiStripThresholdFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripThresholdFakeESSource.cc

 Description: "fake" SiStripThreshold ESProducer - fixed values from configuration for all thresholds (low, high and cluster)

 Implementation:
     Port of SiStripThresholdGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripThresholdFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripThresholdFakeESSource(const edm::ParameterSet&);
  ~SiStripThresholdFakeESSource() override;

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity ) override;

  typedef std::shared_ptr<SiStripThreshold> ReturnType;
  ReturnType produce(const SiStripThresholdRcd&);

private:
  float m_lTh;
  float m_hTh;
  float m_cTh;
  edm::FileInPath m_file;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripThresholdFakeESSource::SiStripThresholdFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripThresholdRcd>();

  m_lTh = iConfig.getParameter<double>("LowTh");
  m_hTh = iConfig.getParameter<double>("HighTh");
  m_cTh = iConfig.getParameter<double>("ClusTh");
  m_file = iConfig.getParameter<edm::FileInPath>("file");
}

SiStripThresholdFakeESSource::~SiStripThresholdFakeESSource() {}

void SiStripThresholdFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripThresholdFakeESSource::ReturnType
SiStripThresholdFakeESSource::produce(const SiStripThresholdRcd& iRecord)
{
  using namespace edm::es;

  std::shared_ptr<SiStripThreshold> threshold{new SiStripThreshold};

  SiStripDetInfoFileReader reader{m_file.fullPath()};

  for ( const auto& elm : reader.getAllData() ) {
    //Generate Thresholds for det detid
    SiStripThreshold::Container theSiStripVector;
    uint16_t strip=0;

    threshold->setData(strip, m_lTh, m_hTh, m_cTh, theSiStripVector);
    LogDebug("SiStripThresholdFakeESSource::produce") << "detid: "  << elm.first << " \t"
						      << "firstStrip: " << strip << " \t" << theSiStripVector.back().getFirstStrip() << " \t"
						      << "lTh: "    << m_lTh     << " \t" << theSiStripVector.back().getLth() << " \t"
						      << "hTh: "    << m_hTh     << " \t" << theSiStripVector.back().getHth() << " \t"
						      << "FirstStrip_and_Hth: " << theSiStripVector.back().FirstStrip_and_Hth << " \t";

    if ( ! threshold->put(elm.first, theSiStripVector) ) {
      edm::LogError("SiStripThresholdFakeESSource::produce ") << " detid already exists";
    }
  }

  return threshold;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripThresholdFakeESSource);
