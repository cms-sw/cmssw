// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripApvGainFakeESSource
//
/**\class SiStripApvGainFakeESSource SiStripApvGainFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripApvGainFakeESSource.cc

 Description: "fake" SiStripApvGain ESProducer - fixed (or gaussian-generated) value from configuration for all gains

 Implementation:
     Port of SiStripApvGainGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

class SiStripApvGainFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripApvGainFakeESSource(const edm::ParameterSet&);
  ~SiStripApvGainFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripApvGain> ReturnType;
  ReturnType produce(const SiStripApvGainRcd&);

private:
  std::string m_genMode;
  double m_meanGain;
  double m_sigmaGain;
  double m_minimumPosValue;
  uint32_t m_printDebug;
  SiStripDetInfo m_detInfo;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandGauss.h"

SiStripApvGainFakeESSource::SiStripApvGainFakeESSource(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
  findingRecord<SiStripApvGainRcd>();

  m_genMode = iConfig.getParameter<std::string>("genMode");
  m_meanGain = iConfig.getParameter<double>("MeanGain");
  m_sigmaGain = iConfig.getParameter<double>("SigmaGain");
  m_minimumPosValue = iConfig.getParameter<double>("MinPositiveGain");
  m_printDebug = iConfig.getUntrackedParameter<uint32_t>("printDebug", 5);
  m_detInfo = SiStripDetInfoFileReader::read(iConfig.getParameter<edm::FileInPath>("SiStripDetInfoFile").fullPath());
}

SiStripApvGainFakeESSource::~SiStripApvGainFakeESSource() {}

void SiStripApvGainFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                const edm::IOVSyncValue& iov,
                                                edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripApvGainFakeESSource::ReturnType SiStripApvGainFakeESSource::produce(const SiStripApvGainRcd& iRecord) {
  using namespace edm::es;

  auto apvGain = std::make_unique<SiStripApvGain>();

  uint32_t count{0};
  for (const auto& elm : m_detInfo.getAllData()) {
    std::vector<float> theSiStripVector;
    for (unsigned short j = 0; j < elm.second.nApvs; ++j) {
      float gainValue;
      if (m_genMode == "default") {
        gainValue = m_meanGain;
      } else if (m_genMode == "gaussian") {
        gainValue = CLHEP::RandGauss::shoot(m_meanGain, m_sigmaGain);
        if (gainValue <= m_minimumPosValue) {
          gainValue = m_minimumPosValue;
        }
      } else {
        LogDebug("SiStripApvGain") << "ERROR: wrong genMode specifier : " << m_genMode
                                   << ", please select one of \"default\" or \"gaussian\"";
        exit(1);
      }

      if (count < m_printDebug) {
        edm::LogInfo("SiStripApvGainGenerator") << "detid: " << elm.first << " Apv: " << j << " gain: " << gainValue;
      }
      theSiStripVector.push_back(gainValue);
    }
    ++count;

    if (!apvGain->put(elm.first, SiStripApvGain::Range{theSiStripVector.begin(), theSiStripVector.end()})) {
      edm::LogError("SiStripApvGainGenerator") << " detid already exists";
    }
  }

  return apvGain;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripApvGainFakeESSource);
