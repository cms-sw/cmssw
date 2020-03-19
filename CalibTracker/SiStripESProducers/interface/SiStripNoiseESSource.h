#ifndef CalibTracker_SiStripESProducers_SiStripNoiseESSource_H
#define CalibTracker_SiStripESProducers_SiStripNoiseESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripNoises;
class SiStripNoisesRcd;

/** 
    @class SiStripNoiseESSource
    @brief Pure virtual class for EventSetup sources of SiStripNoises.
    @author R.Bainbridge
*/
class SiStripNoiseESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripNoiseESSource(const edm::ParameterSet&);
  ~SiStripNoiseESSource() override { ; }

  virtual std::unique_ptr<SiStripNoises> produce(const SiStripNoisesRcd&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  SiStripNoiseESSource(const SiStripNoiseESSource&) = delete;
  const SiStripNoiseESSource& operator=(const SiStripNoiseESSource&) = delete;

  virtual SiStripNoises* makeNoise() = 0;
};

#endif  // CalibTracker_SiStripESProducers_SiStripNoiseESSource_H
