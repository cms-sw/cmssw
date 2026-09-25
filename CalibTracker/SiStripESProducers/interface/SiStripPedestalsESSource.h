#ifndef CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H
#define CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripPedestals;
class SiStripPedestalsRcd;

/** 
    @class SiStripPedestalsESSource
    @brief Pure virtual class for EventSetup sources of SiStripPedestals.
    @author R.Bainbridge
*/
class SiStripPedestalsESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit SiStripPedestalsESSource(const edm::ParameterSet&);
  SiStripPedestalsESSource(const SiStripPedestalsESSource&) = delete;
  const SiStripPedestalsESSource& operator=(const SiStripPedestalsESSource&) = delete;
  ~SiStripPedestalsESSource() override { ; }

  std::unique_ptr<SiStripPedestals> produce(const SiStripPedestalsRcd&);

private:
  virtual SiStripPedestals* makePedestals() = 0;
};

#endif  // CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H
