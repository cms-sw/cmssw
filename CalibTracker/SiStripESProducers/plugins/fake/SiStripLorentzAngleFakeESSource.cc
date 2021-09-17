// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripLorentzAngleFakeESSource
//
/**\class SiStripLorentzAngleFakeESSource SiStripLorentzAngleFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripLorentzAngleFakeESSource.cc

 Description: Generator of the ideal/fake conditions for the LorentzAngle.

 It receives input values with layer granularity and it is able to perform gaussian smearing or
 use a uniform distribution at the module level.
 Depending on the parameters passed via cfg, it is able to generate the values per DetId
 with a gaussian distribution and a uniform distribution. When setting the sigma of the gaussian to 0
 and passing a single value the generated values are fixed.
 For TID and TEC the decision to generate with a uniform distribution comes from the setting
 for the first layers of TIB and TOB.

 Implementation:
     Port of SiStripLorentzAngleGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>
#include <numeric>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class SiStripLorentzAngleFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripLorentzAngleFakeESSource(const edm::ParameterSet&);
  ~SiStripLorentzAngleFakeESSource() override;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripLorentzAngle> ReturnType;
  ReturnType produce(const SiStripLorentzAngleRcd&);

private:
  std::vector<double> m_TIB_EstimatedValuesMin;
  std::vector<double> m_TIB_EstimatedValuesMax;
  std::vector<double> m_TOB_EstimatedValuesMin;
  std::vector<double> m_TOB_EstimatedValuesMax;
  std::vector<double> m_TIB_PerCent_Errs;
  std::vector<double> m_TOB_PerCent_Errs;
  std::vector<double> m_StdDevs_TIB;
  std::vector<double> m_StdDevs_TOB;
  std::vector<bool> m_uniformTIB;
  std::vector<bool> m_uniformTOB;
  double m_TIBmeanValueMin;
  double m_TIBmeanValueMax;
  double m_TOBmeanValueMin;
  double m_TOBmeanValueMax;
  double m_TIBmeanPerCentError;
  double m_TOBmeanPerCentError;
  double m_TIBmeanStdDev;
  double m_TOBmeanStdDev;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> m_geomDetToken;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

namespace {  // helper methods
  /// Method used to determine whether to generate with a uniform distribution for each layer
  void setUniform(const std::vector<double>& estimatedValuesMin,
                  const std::vector<double>& estimatedValuesMax,
                  std::vector<bool>& uniform) {
    if (!estimatedValuesMax.empty()) {
      std::vector<double>::const_iterator min = estimatedValuesMin.begin();
      std::vector<double>::const_iterator max = estimatedValuesMax.begin();
      std::vector<bool>::iterator uniformIt = uniform.begin();
      for (; min != estimatedValuesMin.end(); ++min, ++max, ++uniformIt) {
        if (*min != *max)
          *uniformIt = true;
      }
    }
  }

  double computeSigma(const double& value, const double& perCentError) { return (perCentError / 100) * value; }

  /**
   * Generate a hallMobility value according to the parameters passed in the cfg.
   * - If a min and max value were passed it takes the value from a uniform distribution.
   * - If only a single value was passed and the error is set != 0 it takes the value from a gaussian distribution.
   * - If the error is 0 and only one value is passed it takes the fixed min value.
   */
  float hallMobility(const double& meanMin, const double& meanMax, const double& sigma, const bool uniform) {
    if (uniform) {
      return CLHEP::RandFlat::shoot(meanMin, meanMax);
    } else if (sigma > 0) {
      return CLHEP::RandGauss::shoot(meanMin, sigma);
    } else {
      return meanMin;
    }
  }
}  // namespace

SiStripLorentzAngleFakeESSource::SiStripLorentzAngleFakeESSource(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);
  m_tTopoToken = cc.consumes();
  m_geomDetToken = cc.consumes();
  findingRecord<SiStripLorentzAngleRcd>();

  m_TIB_EstimatedValuesMin = iConfig.getParameter<std::vector<double>>("TIB_EstimatedValuesMin");
  m_TIB_EstimatedValuesMax = iConfig.getParameter<std::vector<double>>("TIB_EstimatedValuesMax");
  m_TOB_EstimatedValuesMin = iConfig.getParameter<std::vector<double>>("TOB_EstimatedValuesMin");
  m_TOB_EstimatedValuesMax = iConfig.getParameter<std::vector<double>>("TOB_EstimatedValuesMax");
  m_TIB_PerCent_Errs = iConfig.getParameter<std::vector<double>>("TIB_PerCent_Errs");
  m_TOB_PerCent_Errs = iConfig.getParameter<std::vector<double>>("TOB_PerCent_Errs");

  // If max values are passed they must be equal in number to the min values.
  if (((!m_TIB_EstimatedValuesMax.empty()) && (m_TIB_EstimatedValuesMin.size() != m_TIB_EstimatedValuesMax.size())) ||
      ((!m_TOB_EstimatedValuesMax.empty()) && (m_TOB_EstimatedValuesMin.size() != m_TOB_EstimatedValuesMax.size()))) {
    std::cout << "ERROR: size of min and max values is different" << std::endl;
    std::cout << "TIB_EstimatedValuesMin.size() = " << m_TIB_EstimatedValuesMin.size()
              << ", TIB_EstimatedValuesMax.size() " << m_TIB_EstimatedValuesMax.size() << std::endl;
    std::cout << "TOB_EstimatedValuesMin.size() = " << m_TOB_EstimatedValuesMin.size()
              << ", TOB_EstimatedValuesMax.size() " << m_TOB_EstimatedValuesMax.size() << std::endl;
  }

  m_uniformTIB = std::vector<bool>(m_TIB_EstimatedValuesMin.size(), false);
  m_uniformTOB = std::vector<bool>(m_TOB_EstimatedValuesMin.size(), false);
  setUniform(m_TIB_EstimatedValuesMin, m_TIB_EstimatedValuesMax, m_uniformTIB);
  setUniform(m_TOB_EstimatedValuesMin, m_TOB_EstimatedValuesMax, m_uniformTOB);

  // Compute standard deviations
  m_StdDevs_TIB = std::vector<double>(m_TIB_EstimatedValuesMin.size(), 0);
  m_StdDevs_TOB = std::vector<double>(m_TOB_EstimatedValuesMin.size(), 0);
  transform(m_TIB_EstimatedValuesMin.begin(),
            m_TIB_EstimatedValuesMin.end(),
            m_TIB_PerCent_Errs.begin(),
            m_StdDevs_TIB.begin(),
            computeSigma);
  transform(m_TOB_EstimatedValuesMin.begin(),
            m_TOB_EstimatedValuesMin.end(),
            m_TOB_PerCent_Errs.begin(),
            m_StdDevs_TOB.begin(),
            computeSigma);

  // Compute mean values to be used with TID and TEC
  m_TIBmeanValueMin = std::accumulate(m_TIB_EstimatedValuesMin.begin(), m_TIB_EstimatedValuesMin.end(), 0.) /
                      double(m_TIB_EstimatedValuesMin.size());
  m_TIBmeanValueMax = std::accumulate(m_TIB_EstimatedValuesMax.begin(), m_TIB_EstimatedValuesMax.end(), 0.) /
                      double(m_TIB_EstimatedValuesMax.size());
  m_TOBmeanValueMin = std::accumulate(m_TOB_EstimatedValuesMin.begin(), m_TOB_EstimatedValuesMin.end(), 0.) /
                      double(m_TOB_EstimatedValuesMin.size());
  m_TOBmeanValueMax = std::accumulate(m_TOB_EstimatedValuesMax.begin(), m_TOB_EstimatedValuesMax.end(), 0.) /
                      double(m_TOB_EstimatedValuesMax.size());
  m_TIBmeanPerCentError =
      std::accumulate(m_TIB_PerCent_Errs.begin(), m_TIB_PerCent_Errs.end(), 0.) / double(m_TIB_PerCent_Errs.size());
  m_TOBmeanPerCentError =
      std::accumulate(m_TOB_PerCent_Errs.begin(), m_TOB_PerCent_Errs.end(), 0.) / double(m_TOB_PerCent_Errs.size());
  m_TIBmeanStdDev = (m_TIBmeanPerCentError / 100) * m_TIBmeanValueMin;
  m_TOBmeanStdDev = (m_TOBmeanPerCentError / 100) * m_TOBmeanValueMin;
}

SiStripLorentzAngleFakeESSource::~SiStripLorentzAngleFakeESSource() {}

void SiStripLorentzAngleFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                     const edm::IOVSyncValue& iov,
                                                     edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripLorentzAngleFakeESSource::ReturnType SiStripLorentzAngleFakeESSource::produce(
    const SiStripLorentzAngleRcd& iRecord) {
  using namespace edm::es;

  const auto& geomDet = iRecord.getRecord<TrackerTopologyRcd>().get(m_geomDetToken);
  const auto& tTopo = iRecord.get(m_tTopoToken);

  auto lorentzAngle = std::make_unique<SiStripLorentzAngle>();

  for (const auto detId : TrackerGeometryUtils::getSiStripDetIds(geomDet)) {
    const DetId detectorId = DetId(detId);
    const int subDet = detectorId.subdetId();

    float mobi{0.};

    if (subDet == int(StripSubdetector::TIB)) {
      const int layerId = tTopo.tibLayer(detectorId) - 1;
      mobi = hallMobility(m_TIB_EstimatedValuesMin[layerId],
                          m_TIB_EstimatedValuesMax[layerId],
                          m_StdDevs_TIB[layerId],
                          m_uniformTIB[layerId]);
    } else if (subDet == int(StripSubdetector::TOB)) {
      const int layerId = tTopo.tobLayer(detectorId) - 1;
      mobi = hallMobility(m_TOB_EstimatedValuesMin[layerId],
                          m_TOB_EstimatedValuesMax[layerId],
                          m_StdDevs_TOB[layerId],
                          m_uniformTOB[layerId]);
    } else if (subDet == int(StripSubdetector::TID)) {
      // ATTENTION: as of now the uniform generation for TID is decided by the setting for layer 0 of TIB
      mobi = hallMobility(m_TIBmeanValueMin, m_TIBmeanValueMax, m_TIBmeanStdDev, m_uniformTIB[0]);
    }
    if (subDet == int(StripSubdetector::TEC)) {
      if (tTopo.tecRing(detectorId) < 5) {
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TIB
        mobi = hallMobility(m_TIBmeanValueMin, m_TIBmeanValueMax, m_TIBmeanStdDev, m_uniformTIB[0]);
      } else {
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TOB
        mobi = hallMobility(m_TOBmeanValueMin, m_TOBmeanValueMax, m_TOBmeanStdDev, m_uniformTOB[0]);
      }
    }

    if (!lorentzAngle->putLorentzAngle(detId, mobi)) {
      edm::LogError("SiStripLorentzAngleFakeESSource::produce ") << " detid already exists";
    }
  }

  return lorentzAngle;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripLorentzAngleFakeESSource);
