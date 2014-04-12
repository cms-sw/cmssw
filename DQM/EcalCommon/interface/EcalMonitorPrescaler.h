#ifndef EcalMonitorPrescaler_H
#define EcalMonitorPrescaler_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include <utility>

class EcalMonitorPrescaler: public edm::EDFilter {
 public:
  EcalMonitorPrescaler(edm::ParameterSet const&);
  ~EcalMonitorPrescaler();

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  bool filter(edm::Event&, edm::EventSetup const&) override;

 private:
  enum Prescalers {
    kPhysics,
    kCosmics,
    kCalibration,
    kLaser,
    kLed,
    kTestPulse,
    kPedestal,
    nPrescalers
  };

  static uint32_t filterBits_[nPrescalers];

  edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;

  std::pair<unsigned, unsigned> prescalers_[nPrescalers];
};

#endif // EcalMonitorPrescaler_H
