#ifndef SiStripNoiseNormalizedWithApvGainBuilder_H
#define SiStripNoiseNormalizedWithApvGainBuilder_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class TrackerTopology;
class TrackerGeometry;
/**
 * Produces a noise tag using the same settings as the service used in the DummyDBWriter, but
 * it receives a SiStripApvGain tag from the EventSetup and uses the gain values (per apv) to
 * rescale the noise (per strip).
 */

class SiStripNoiseNormalizedWithApvGainBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripNoiseNormalizedWithApvGainBuilder(const edm::ParameterSet& iConfig);

  ~SiStripNoiseNormalizedWithApvGainBuilder() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  /// Fills the parameters read from cfg and matching the name in the given map
  void fillParameters(std::map<int, std::vector<double> >& mapToFill, const std::string& parameterName) const;
  /**
   * Fills the map with the paramters for the given subdetector. <br>
   * Each vector "v" holds the parameters for the layers/rings, if the vector has only one parameter
   * all the layers/rings get that parameter. <br>
   * The only other possibility is that the number of parameters equals the number of layers, otherwise
   * an exception of type "Configuration" will be thrown.
   */
  void fillSubDetParameter(std::map<int, std::vector<double> >& mapToFill,
                           const std::vector<double>& v,
                           const int subDet,
                           const unsigned short layers) const;

  inline void printLog(const uint32_t detId, const unsigned short strip, const double& noise) const {
    edm::LogInfo("SiStripNoisesDummyCalculator")
        << "detid: " << detId << " strip: " << strip << " noise: " << noise << " \t" << std::endl;
  }

  edm::FileInPath fp_;
  bool printdebug_;
  edm::ParameterSet pset_;

  double electronsPerADC_;
  double minimumPosValue_;
  bool stripLengthMode_;
  uint32_t printDebug_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> inputApvGainToken_;
};

#endif
