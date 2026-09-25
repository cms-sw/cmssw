#ifndef CalibTracker_SiStripESProducers_SiStripGainESSource_H
#define CalibTracker_SiStripESProducers_SiStripGainESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripApvGain;
class SiStripApvGainRcd;

/** 
    @class SiStripGainESSource
    @brief Pure virtual class for EventSetup sources of SiStripApvGain.
    @author R.Bainbridge
*/
class SiStripGainESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit SiStripGainESSource(const edm::ParameterSet&);
  SiStripGainESSource(const SiStripGainESSource&) = delete;
  const SiStripGainESSource& operator=(const SiStripGainESSource&) = delete;
  ~SiStripGainESSource() override { ; }

  std::unique_ptr<SiStripApvGain> produce(const SiStripApvGainRcd&);

private:
  virtual SiStripApvGain* makeGain() = 0;
};

#endif  // CalibTracker_SiStripESProducers_SiStripGainESSource_H
