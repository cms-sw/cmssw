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

class SiStripNoisesFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripNoisesFakeESSource(const edm::ParameterSet&);
  ~SiStripNoisesFakeESSource();

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity );

  typedef std::shared_ptr<SiStripNoises> ReturnType;
  ReturnType produce(const SiStripNoisesRcd&);

private:
  bool m_stripLengthMode;
  double m_noisePar0;
  std::map<int, std::vector<double>> m_noisePar1;
  std::map<int, std::vector<double>> m_noisePar2;
  edm::FileInPath m_file;
  uint32_t m_printDebug;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CLHEP/Random/RandGauss.h"

namespace { // helper methods
  inline void printLog(const uint32_t detId, const unsigned short strip, const double & noise)
  {
    edm::LogInfo("SiStripNoisesDummyCalculator") << "detid: " << detId << " strip: " << strip <<  " noise: " << noise;
  }

  /**
   * Fills the map with the paramters for the given subdetector. <br>
   * Each vector "v" holds the parameters for the layers/rings, if the vector has only one parameter
   * all the layers/rings get that parameter. <br>
   * The only other possibility is that the number of parameters equals the number of layers, otherwise
   * an exception of type "Configuration" will be thrown.
   */
  void fillSubDetParameter( std::map<int,std::vector<double>>& mapToFill, const std::vector<double>& v, const int subDet, const unsigned short layers )
  {
    if ( v.size() == layers ) {
      mapToFill.insert(std::make_pair(subDet, v));
    } else if ( v.size() == 1 ) {
      std::vector<double> parV(layers, v[0]);
      mapToFill.insert(std::make_pair(subDet, parV));
    } else {
      throw cms::Exception("Configuration") << "ERROR: number of parameters for subDet " << subDet << " are " << v.size() << ". They must be either 1 or " << layers << std::endl;
    }
  }
  /// Fills the parameters read from cfg and matching the name in the given map
  void fillParameters( std::map<int,std::vector<double>>& mapToFill, const edm::ParameterSet& pset, const std::string& parameterName )
  {
    fillSubDetParameter(mapToFill, pset.getParameter<std::vector<double>>(parameterName+"TIB"), int(StripSubdetector::TIB), 4); // 4 TIB layers
    fillSubDetParameter(mapToFill, pset.getParameter<std::vector<double>>(parameterName+"TID"), int(StripSubdetector::TID), 3); // TID rings
    fillSubDetParameter(mapToFill, pset.getParameter<std::vector<double>>(parameterName+"TOB"), int(StripSubdetector::TOB), 6); // TOB layers
    fillSubDetParameter(mapToFill, pset.getParameter<std::vector<double>>(parameterName+"TEC"), int(StripSubdetector::TEC), 7); // TEC rings
  }


  /// Given the map and the detid it returns the corresponding layer/ring
  std::pair<int, int> subDetAndLayer(const uint32_t detId, const TrackerTopology* tTopo)
  {
    int layerId = 0;

    const DetId detectorId=DetId(detId);
    const int subDet = detectorId.subdetId();

    if ( subDet == int(StripSubdetector::TIB) ) {
      layerId = tTopo->tibLayer(detectorId) - 1;
    } else if ( subDet == int(StripSubdetector::TOB) ) {
      layerId = tTopo->tobLayer(detectorId) - 1;
    } else if ( subDet == int(StripSubdetector::TID) ) {
      layerId = tTopo->tidRing(detectorId) - 1;
    }
    if (subDet == int(StripSubdetector::TEC)) {
      layerId = tTopo->tecRing(detectorId) - - 1;
    }
    return std::make_pair(subDet, layerId);
  }
}

SiStripNoisesFakeESSource::SiStripNoisesFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripNoisesRcd>();

  m_stripLengthMode = iConfig.getParameter<bool>("StripLengthMode");

  if ( ! m_stripLengthMode ) {
    //parameters for random noise generation. not used if Strip length mode is chosen
    m_noisePar0 = iConfig.getParameter<double>("MinPositiveNoise");
    fillParameters(m_noisePar1, iConfig, "MeanNoise");
    fillParameters(m_noisePar2, iConfig, "SigmaNoise");
  } else {
    //parameters for strip length proportional noise generation. not used if random mode is chosen
    m_noisePar0 = iConfig.getParameter<double>("electronPerAdc");
    fillParameters(m_noisePar1, iConfig, "NoiseStripLengthSlope");
    fillParameters(m_noisePar2, iConfig, "NoiseStripLengthQuote");
  }

  m_file = iConfig.getParameter<edm::FileInPath>("file");
  m_printDebug = iConfig.getUntrackedParameter<uint32_t>("printDebug", 5);
}

SiStripNoisesFakeESSource::~SiStripNoisesFakeESSource() {}

void SiStripNoisesFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripNoisesFakeESSource::ReturnType
SiStripNoisesFakeESSource::produce(const SiStripNoisesRcd& iRecord)
{
  using namespace edm::es;

  edm::ESHandle<TrackerTopology> tTopo;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopo);

  std::shared_ptr<SiStripNoises> noises{new SiStripNoises};

  SiStripDetInfoFileReader reader{m_file.fullPath()};

  uint32_t count{0};
  for ( const auto& elm : reader.getAllData() ) {
    //Generate Noises for det detid
    SiStripNoises::InputVector theSiStripVector;
    std::pair<int, int> sl = subDetAndLayer(elm.first, tTopo.product());

    if ( m_stripLengthMode ) {
      // Use strip length
      const double linearSlope{m_noisePar1[sl.first][sl.second]};
      const double linearQuote{m_noisePar2[sl.first][sl.second]};
      const double stripLength{elm.second.stripLength};
      for ( unsigned short j{0}; j < 128*elm.second.nApvs; ++j ) {
        const float noise = (linearSlope*stripLength + linearQuote) / m_noisePar0;
        if ( count < m_printDebug ) printLog(elm.first, j, noise);
        noises->setData(noise, theSiStripVector);
      }
    } else {
      // Use random generator
      const double meanN {m_noisePar1[sl.first][sl.second]};
      const double sigmaN{m_noisePar2[sl.first][sl.second]};
      for ( unsigned short j{0}; j < 128*elm.second.nApvs; ++j ) {
        const float noise = std::max(CLHEP::RandGauss::shoot(meanN, sigmaN), m_noisePar0);
        if ( count < m_printDebug ) printLog(elm.first, j, noise);
        noises->setData(noise, theSiStripVector);
      }
    }
    ++count;

    if ( ! noises->put(elm.first, theSiStripVector) ) {
      edm::LogError("SiStripNoisesFakeESSource::produce ") << " detid already exists";
    }
  }

  return noises;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripNoisesFakeESSource);
