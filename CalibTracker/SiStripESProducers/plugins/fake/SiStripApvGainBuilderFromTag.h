#ifndef SiStripApvGainBuilderFromTag_H
#define SiStripApvGainBuilderFromTag_H

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
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class TrackerTopology;
class TrackerGeometry;
/**
 * Produces a tag for the apvGain object starting from an already existing tag in the db. <br>
 * The input tag is used as starting point and all the values for missing apvs are filled with
 * the meanGain specified via cfg. <br>
 * It will optionally apply a correction specified via cfg if the applyTuning bool is true. <br>
 * The resulting new gain values can optionally be smeared with a gaussian of sigma specified in the cfg. <br>
 * Note: we are not using the usual DummyCondDBWriter because we cannot pass an input tag to a service without
 * several changes to the base classes used together with it.
 */

class SiStripApvGainBuilderFromTag : public edm::EDAnalyzer {
public:
  explicit SiStripApvGainBuilderFromTag(const edm::ParameterSet& iConfig);

  ~SiStripApvGainBuilderFromTag() override{};

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

  bool printdebug_;
  edm::ParameterSet pset_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> inputApvGainToken_;
};

#endif
