#ifndef DQM_EcalMonitorTasks_GpuTask_H
#define DQM_EcalMonitorTasks_GpuTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class GpuTask : public DQWorkerTask {
  public:
    GpuTask();
    ~GpuTask() override {}

    void addDependencies(DependencySet&) override;

    bool filterRunType(short const*) override;

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    bool analyze(void const*, Collections) override;

    template <typename DigiCollection>
    void runOnCpuDigis(DigiCollection const&, Collections);
    template <typename DigiCollection>
    void runOnGpuDigis(DigiCollection const&, Collections);
    void runOnCpuUncalibRecHits(EcalUncalibratedRecHitCollection const&, Collections);
    void runOnGpuUncalibRecHits(EcalUncalibratedRecHitCollection const&, Collections);
    void runOnCpuRecHits(EcalRecHitCollection const&, Collections);
    void runOnGpuRecHits(EcalRecHitCollection const&, Collections);

  private:
    void setParams(edm::ParameterSet const&) override;

    bool runGpuTask_;
    bool gpuOnlyPlots_;
    std::vector<int> uncalibOOTAmps_;

    // Static cast to EB/EEDigiCollection when using
    // Defined as void pointers to make compiler happy
    void const* EBCpuDigis_;
    void const* EECpuDigis_;

    EcalUncalibratedRecHitCollection const* EBCpuUncalibRecHits_;
    EcalUncalibratedRecHitCollection const* EECpuUncalibRecHits_;

    EcalRecHitCollection const* EBCpuRecHits_;
    EcalRecHitCollection const* EECpuRecHits_;
  };

  inline bool GpuTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBCpuDigi:
        if (_p && runGpuTask_)
          runOnCpuDigis(*static_cast<EBDigiCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEECpuDigi:
        if (_p && runGpuTask_)
          runOnCpuDigis(*static_cast<EEDigiCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEBGpuDigi:
        if (_p && runGpuTask_)
          runOnGpuDigis(*static_cast<EBDigiCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEEGpuDigi:
        if (_p && runGpuTask_)
          runOnGpuDigis(*static_cast<EEDigiCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEBCpuUncalibRecHit:
      case kEECpuUncalibRecHit:
        if (_p && runGpuTask_)
          runOnCpuUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEBGpuUncalibRecHit:
      case kEEGpuUncalibRecHit:
        if (_p && runGpuTask_)
          runOnGpuUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEBCpuRecHit:
      case kEECpuRecHit:
        if (_p && runGpuTask_)
          runOnCpuRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      case kEBGpuRecHit:
      case kEEGpuRecHit:
        if (_p && runGpuTask_)
          runOnGpuRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return runGpuTask_;
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
