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

#include <memory>

class ClusterizerUnitTesterESProducer : public edm::ESProducer {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;

public:
  ClusterizerUnitTesterESProducer(const PSet&);
  ~ClusterizerUnitTesterESProducer() {}
  std::shared_ptr<const SiStripGain> produceGainRcd(const SiStripGainRcd&) { return gain_; }
  std::shared_ptr<const SiStripNoises> produceNoisesRcd(const SiStripNoisesRcd&) { return noises_; }
  std::shared_ptr<const SiStripQuality> produceQualityRcd(const SiStripQualityRcd&) { return quality_; }

private:
  void extractNoiseGainQuality(const PSet&, SiStripQuality*, SiStripApvGain*, SiStripNoises*);
  void extractNoiseGainQualityForDetId(uint32_t, const VPSet&, SiStripQuality*, SiStripApvGain*, SiStripNoises*);

  void setNoises(uint32_t, std::vector<std::pair<uint16_t, float> >&, SiStripNoises*);
  void setGains(uint32_t, std::vector<std::pair<uint16_t, float> >&, SiStripApvGain*);

  // These objects might be shared across multiple concurrent
  // IOVs and are not allowed to be modified after the module
  // constructor finishes.
  std::unique_ptr<const SiStripApvGain> apvGain_;
  std::shared_ptr<const SiStripGain> gain_;
  std::shared_ptr<const SiStripNoises> noises_;
  std::shared_ptr<const SiStripQuality> quality_;
};
#endif
