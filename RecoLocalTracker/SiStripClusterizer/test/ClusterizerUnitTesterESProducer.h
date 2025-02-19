#ifndef ClusterizerUnitTestESProducer_h
#define ClusterizerUnitTestESProducer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <boost/shared_ptr.hpp>

class ClusterizerUnitTesterESProducer: public edm::ESProducer {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;
 public:
  ClusterizerUnitTesterESProducer(const PSet&);
  ~ClusterizerUnitTesterESProducer(){}
  boost::shared_ptr<SiStripGain> produceGainRcd(const SiStripGainRcd&) { return gain;}
  boost::shared_ptr<SiStripNoises> produceNoisesRcd(const SiStripNoisesRcd&) { return noises;}
  boost::shared_ptr<SiStripQuality> produceQualityRcd(const SiStripQualityRcd&) {return quality;}
 private:
  
  void extractNoiseGainQuality(const PSet&);
  void extractNoiseGainQualityForDetId(uint32_t, const VPSet&);

  void setNoises(uint32_t, std::vector<std::pair<uint16_t,float> >&);
  void setGains( uint32_t, std::vector<std::pair<uint16_t,float> >&);

  boost::shared_ptr<SiStripApvGain> apvGain;
  boost::shared_ptr<SiStripGain> gain;
  boost::shared_ptr<SiStripNoises>  noises;
  boost::shared_ptr<SiStripQuality> quality;
};
#endif
