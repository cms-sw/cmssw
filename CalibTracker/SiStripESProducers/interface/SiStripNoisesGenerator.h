#ifndef CalibTracker_SiStripESProducers_SiStripNoisesGenerator_H
#define CalibTracker_SiStripESProducers_SiStripNoisesGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripDepCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include <string>
#include <map>

class SiStripNoisesGenerator : public SiStripDepCondObjBuilderBase<SiStripNoises,TrackerTopology> {
 public:

  explicit SiStripNoisesGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripNoisesGenerator();
  
  void getObj(SiStripNoises* & obj, const TrackerTopology* tTopo){obj=createObject(tTopo);}

 private:

  SiStripNoises*  createObject(const TrackerTopology* tTopo);
  /// Given the map and the detid it returns the corresponding layer/ring
  std::pair<int, int> subDetAndLayer(const uint32_t detit, const TrackerTopology* tTopo) const;
  /// Fills the parameters read from cfg and matching the name in the given map
  void fillParameters(std::map<int, std::vector<double> > & mapToFill, const std::string & parameterName) const;
  /**
   * Fills the map with the paramters for the given subdetector. <br>
   * Each vector "v" holds the parameters for the layers/rings, if the vector has only one parameter
   * all the layers/rings get that parameter. <br>
   * The only other possibility is that the number of parameters equals the number of layers, otherwise
   * an exception of type "Configuration" will be thrown.
   */
  void fillSubDetParameter(std::map<int, std::vector<double> > & mapToFill, const std::vector<double> & v, const int subDet, const unsigned short layers) const;

  inline void printLog(const uint32_t detId, const unsigned short strip, const double & noise) const
  {
    edm::LogInfo("SiStripNoisesDummyCalculator") << "detid: " << detId << " strip: " << strip <<  " noise: " << noise     << " \t"   << std::endl;
  }

  double electronsPerADC_;
  double minimumPosValue_;
  bool stripLengthMode_;
  uint32_t printDebug_;
};

#endif 
