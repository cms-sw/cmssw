#ifndef CalibTracker_RPCCalibration_RPCPerformanceESSource_H
#define CalibTracker_RPCCalibration_RPCPerformanceESSource_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class RPCStripNoises;
class RPCStripNoisesRcd;

/**
    @class RPCPerformanceESSource
    @brief Pure virtual class for EventSetup sources of RPCStripNoises.
    @author R. Trentadue
*/
class RPCPerformanceESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit RPCPerformanceESSource(const edm::ParameterSet &);
  ~RPCPerformanceESSource() override { ; }

  std::unique_ptr<RPCStripNoises> produce(const RPCStripNoisesRcd &);

  // private:

  RPCPerformanceESSource(const RPCPerformanceESSource &) = delete;
  const RPCPerformanceESSource &operator=(const RPCPerformanceESSource &) = delete;

  virtual RPCStripNoises *makeNoise() = 0;
};

#endif  // CalibTracker_RPCCalibration_RPCPerformanceESSource_H
