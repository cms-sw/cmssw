// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripNoisesFakeESSource
//
/**\class SiStripNoisesFakeESSource SiStripNoisesFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripNoisesFakeESSource.cc

 Description: "fake" SiStripNoises ESProducer
 - strip length mode: noise is a linear function of strip length (with different parameters for each layer - the same for all if only one is given)
 - random mode: noise of each strip is generated from a Gaussian distribution (with different parameters for each layer - the same for all if only one is given)

 Implementation:
     Port of SiStripNoisesGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SiStripFakeAPVParameters.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

class SiStripNoisesFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripNoisesFakeESSource(const edm::ParameterSet&);
  ~SiStripNoisesFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripNoises> ReturnType;
  ReturnType produce(const SiStripNoisesRcd&);

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
  bool m_stripLengthMode;
  double m_noisePar0;
  SiStripFakeAPVParameters m_noisePar1;
  SiStripFakeAPVParameters m_noisePar2;
  uint32_t m_printDebug;
  SiStripDetInfo m_detInfo;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CLHEP/Random/RandGauss.h"

namespace {  // helper methods
  inline void printLog(const uint32_t detId, const unsigned short strip, const double& noise) {
    edm::LogInfo("SiStripNoisesDummyCalculator") << "detid: " << detId << " strip: " << strip << " noise: " << noise;
  }
}  // namespace

SiStripNoisesFakeESSource::SiStripNoisesFakeESSource(const edm::ParameterSet& iConfig)
    : m_tTopoToken(setWhatProduced(this).consumes()) {
  findingRecord<SiStripNoisesRcd>();

  m_stripLengthMode = iConfig.getParameter<bool>("StripLengthMode");

  if (!m_stripLengthMode) {
    //parameters for random noise generation. not used if Strip length mode is chosen
    m_noisePar0 = iConfig.getParameter<double>("MinPositiveNoise");
    m_noisePar1 = SiStripFakeAPVParameters(iConfig, "MeanNoise");
    m_noisePar2 = SiStripFakeAPVParameters(iConfig, "SigmaNoise");
  } else {
    //parameters for strip length proportional noise generation. not used if random mode is chosen
    m_noisePar0 = iConfig.getParameter<double>("electronPerAdc");
    m_noisePar1 = SiStripFakeAPVParameters(iConfig, "NoiseStripLengthSlope");
    m_noisePar2 = SiStripFakeAPVParameters(iConfig, "NoiseStripLengthQuote");
  }

  m_printDebug = iConfig.getUntrackedParameter<uint32_t>("printDebug", 5);

  m_detInfo = SiStripDetInfoFileReader::read(iConfig.getParameter<edm::FileInPath>("SiStripDetInfoFile").fullPath());
}

SiStripNoisesFakeESSource::~SiStripNoisesFakeESSource() {}

void SiStripNoisesFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                               const edm::IOVSyncValue& iov,
                                               edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripNoisesFakeESSource::ReturnType SiStripNoisesFakeESSource::produce(const SiStripNoisesRcd& iRecord) {
  using namespace edm::es;

  const auto& tTopo = iRecord.get(m_tTopoToken);

  auto noises = std::make_unique<SiStripNoises>();

  uint32_t count{0};
  for (const auto& elm : m_detInfo.getAllData()) {
    //Generate Noises for det detid
    SiStripNoises::InputVector theSiStripVector;
    SiStripFakeAPVParameters::index sl = SiStripFakeAPVParameters::getIndex(&tTopo, elm.first);

    if (m_stripLengthMode) {
      // Use strip length
      const double linearSlope{m_noisePar1.get(sl)};
      const double linearQuote{m_noisePar2.get(sl)};
      const double stripLength{elm.second.stripLength};
      for (unsigned short j{0}; j < 128 * elm.second.nApvs; ++j) {
        const float noise = (linearSlope * stripLength + linearQuote) / m_noisePar0;
        if (count < m_printDebug)
          printLog(elm.first, j, noise);
        noises->setData(noise, theSiStripVector);
      }
    } else {
      // Use random generator
      const double meanN{m_noisePar1.get(sl)};
      const double sigmaN{m_noisePar2.get(sl)};
      for (unsigned short j{0}; j < 128 * elm.second.nApvs; ++j) {
        const float noise = std::max(CLHEP::RandGauss::shoot(meanN, sigmaN), m_noisePar0);
        if (count < m_printDebug)
          printLog(elm.first, j, noise);
        noises->setData(noise, theSiStripVector);
      }
    }
    ++count;

    if (!noises->put(elm.first, theSiStripVector)) {
      edm::LogError("SiStripNoisesFakeESSource::produce ") << " detid already exists";
    }
  }

  return noises;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripNoisesFakeESSource);
