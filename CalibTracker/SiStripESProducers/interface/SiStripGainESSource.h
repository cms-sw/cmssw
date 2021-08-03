#ifndef CalibTracker_SiStripESProducers_SiStripGainESSource_H
#define CalibTracker_SiStripESProducers_SiStripGainESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripApvGain;
class SiStripApvGainRcd;

/** 
    @class SiStripGainESSource
    @brief Pure virtual class for EventSetup sources of SiStripApvGain.
    @author R.Bainbridge
*/
class SiStripGainESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripGainESSource(const edm::ParameterSet&);
  SiStripGainESSource(const SiStripGainESSource&) = delete;
  const SiStripGainESSource& operator=(const SiStripGainESSource&) = delete;
  ~SiStripGainESSource() override { ; }

  virtual std::unique_ptr<SiStripApvGain> produce(const SiStripApvGainRcd&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  virtual SiStripApvGain* makeGain() = 0;
};

#endif  // CalibTracker_SiStripESProducers_SiStripGainESSource_H
