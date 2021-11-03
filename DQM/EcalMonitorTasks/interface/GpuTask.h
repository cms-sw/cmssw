#ifndef DQM_EcalMonitorTasks_GpuTask_H
#define DQM_EcalMonitorTasks_GpuTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class GpuTask : public DQWorkerTask {
  public:
    GpuTask();
    ~GpuTask() override {}

    void addDependencies(DependencySet&) override;

    bool filterRunType(short const*) override;

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    bool analyze(void const*, Collections) override;

    void runOnCpuRecHits(EcalRecHitCollection const&, Collections);
    void runOnGpuRecHits(EcalRecHitCollection const&, Collections);

  private:
    void setParams(edm::ParameterSet const&) override;

    bool runOnGpu_;
    EcalRecHitCollection const* EBCpuRecHits_;
    EcalRecHitCollection const* EECpuRecHits_;
  };

  inline bool GpuTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBCpuRecHit:
      case kEECpuRecHit:
        if (_p)
          runOnCpuRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return true;
        break;
      case kEBGpuRecHit:
      case kEEGpuRecHit:
        if (_p && runOnGpu_)
          runOnGpuRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return runOnGpu_;
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
