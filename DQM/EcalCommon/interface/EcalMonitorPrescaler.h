#ifndef EcalMonitorPrescaler_H
#define EcalMonitorPrescaler_H

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include <utility>
#include <atomic>

namespace ecaldqm {
  enum Prescalers { kPhysics, kCosmics, kCalibration, kLaser, kLed, kTestPulse, kPedestal, nPrescalers };

  struct PrescaleCounter {
    PrescaleCounter() {
      for (unsigned iP(0); iP != nPrescalers; ++iP)
        counters_[iP] = 0;
    }
    mutable std::atomic<unsigned> counters_[nPrescalers];
  };
}  // namespace ecaldqm

class EcalMonitorPrescaler : public edm::global::EDFilter<edm::RunCache<ecaldqm::PrescaleCounter>> {
public:
  EcalMonitorPrescaler(edm::ParameterSet const &);
  ~EcalMonitorPrescaler() override;

  std::shared_ptr<ecaldqm::PrescaleCounter> globalBeginRun(edm::Run const &, edm::EventSetup const &) const override;
  bool filter(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;
  void globalEndRun(edm::Run const &, edm::EventSetup const &) const override;

private:
  static const uint32_t filterBits_[ecaldqm::nPrescalers];

  edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;

  unsigned prescalers_[ecaldqm::nPrescalers];
};

#endif  // EcalMonitorPrescaler_H
