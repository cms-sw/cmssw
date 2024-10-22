#ifndef IntegrityTask_H
#define IntegrityTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

namespace ecaldqm {

  class IntegrityTask : public DQWorkerTask {
  public:
    IntegrityTask();
    ~IntegrityTask() override {}

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;

    bool analyze(void const*, Collections) override;

    template <class C>
    void runOnDetIdCollection(C const&, Collections);
    void runOnElectronicsIdCollection(EcalElectronicsIdCollection const&, Collections);
  };

  inline bool IntegrityTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBGainErrors:
      case kEBChIdErrors:
      case kEBGainSwitchErrors:
        if (_p)
          runOnDetIdCollection(*static_cast<EBDetIdCollection const*>(_p), _collection);
        return true;
      case kEEGainErrors:
      case kEEChIdErrors:
      case kEEGainSwitchErrors:
        if (_p)
          runOnDetIdCollection(*static_cast<EEDetIdCollection const*>(_p), _collection);
        return true;
        break;
      case kTowerIdErrors:
      case kBlockSizeErrors:
        if (_p)
          runOnElectronicsIdCollection(*static_cast<EcalElectronicsIdCollection const*>(_p), _collection);
        return true;
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
