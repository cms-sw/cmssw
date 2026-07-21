#ifndef DQM_EcalMonitorTasks_EnergyTask_h
#define DQM_EcalMonitorTasks_EnergyTask_h

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class EnergyTask : public DQWorkerTask {
  public:
    EnergyTask();
    ~EnergyTask() override {}

    bool filterRunType(short const*) override;

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    bool analyze(void const*, Collections) override;

    void runOnRecHits(EcalRecHitCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    bool isPhysicsRun_;
    //    float threshS9_;
    bool doEndcaps_;
  };

  inline bool EnergyTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBRecHit:
        if (_p)
          runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p));
        return true;
        break;
      case kEERecHit:
        if (doEndcaps_) {
          if (_p)
            runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p));
          return true;
        }
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
