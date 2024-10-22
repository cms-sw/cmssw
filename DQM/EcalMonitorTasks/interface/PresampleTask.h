#ifndef PresampleTask_H
#define PresampleTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace ecaldqm {
  class PresampleTask : public DQWorkerTask {
  public:
    PresampleTask();
    ~PresampleTask() override {}

    bool filterRunType(short const*) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    bool analyze(void const*, Collections) override;

    template <typename DigiCollection>
    void runOnDigis(DigiCollection const&);
    void setTokens(edm::ConsumesCollector&) override;

  private:
    void setParams(edm::ParameterSet const&) override;
    edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> Pedtoken_;
    bool doPulseMaxCheck_;
    int pulseMaxPosition_;
    int nSamples_;
    MESet* mePedestalByLS;
    bool FillPedestal = false;
  };

  inline bool PresampleTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBDigi:
        if (_p)
          runOnDigis(*static_cast<EBDigiCollection const*>(_p));
        return true;
      case kEEDigi:
        if (_p)
          runOnDigis(*static_cast<EEDigiCollection const*>(_p));
        return true;
        break;
      default:
        break;
    }

    return false;
  }
}  // namespace ecaldqm

#endif
