#ifndef CalibTracker_RPCCalibration_RPCPerformanceESSource_H
#define CalibTracker_RPCCalibration_RPCPerformanceESSource_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class RPCStripNoises;
class RPCStripNoisesRcd;

/**
    @class RPCPerformanceESSource
    @brief Pure virtual class for EventSetup sources of RPCStripNoises.
    @author R. Trentadue
*/
class RPCPerformanceESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  RPCPerformanceESSource(const edm::ParameterSet &);
  ~RPCPerformanceESSource() override { ; }

  std::unique_ptr<RPCStripNoises> produce(const RPCStripNoisesRcd &);

  // protected:

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  // private:

  RPCPerformanceESSource(const RPCPerformanceESSource &);
  const RPCPerformanceESSource &operator=(const RPCPerformanceESSource &);

  virtual RPCStripNoises *makeNoise() = 0;
};

#endif  // CalibTracker_RPCCalibration_RPCPerformanceESSource_H
