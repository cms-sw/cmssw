#ifndef DQM_EcalMonitorTasks_GpuTask_H
#define DQM_EcalMonitorTasks_GpuTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class GpuTask : public DQWorkerTask {
  public:
    GpuTask();
    ~GpuTask() override = default;

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
    bool enableDigi_;
    bool enableUncalib_;
    bool enableRecHit_;

    bool digi1D_;
    bool digi2D_;
    bool uncalib1D_;
    bool uncalib2D_;
    bool rechit1D_;
    bool rechit2D_;

    std::vector<int> uncalibOOTAmps_;

    EBDigiCollection const* EBCpuDigis_;
    EEDigiCollection const* EECpuDigis_;

    EcalUncalibratedRecHitCollection const* EBCpuUncalibRecHits_;
    EcalUncalibratedRecHitCollection const* EECpuUncalibRecHits_;

    EcalRecHitCollection const* EBCpuRecHits_;
    EcalRecHitCollection const* EECpuRecHits_;
  };

  inline bool GpuTask::analyze(void const* collection_data, Collections collection) {
    switch (collection) {
      case kEBCpuDigi:
        if (collection_data && runGpuTask_ && enableDigi_)
          runOnCpuDigis(*static_cast<EBDigiCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEECpuDigi:
        if (collection_data && runGpuTask_ && enableDigi_)
          runOnCpuDigis(*static_cast<EEDigiCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEBGpuDigi:
        if (collection_data && runGpuTask_ && enableDigi_)
          runOnGpuDigis(*static_cast<EBDigiCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEEGpuDigi:
        if (collection_data && runGpuTask_ && enableDigi_)
          runOnGpuDigis(*static_cast<EEDigiCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEBCpuUncalibRecHit:
      case kEECpuUncalibRecHit:
        if (collection_data && runGpuTask_ && enableUncalib_)
          runOnCpuUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEBGpuUncalibRecHit:
      case kEEGpuUncalibRecHit:
        if (collection_data && runGpuTask_ && enableUncalib_)
          runOnGpuUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEBCpuRecHit:
      case kEECpuRecHit:
        if (collection_data && runGpuTask_ && enableRecHit_)
          runOnCpuRecHits(*static_cast<EcalRecHitCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      case kEBGpuRecHit:
      case kEEGpuRecHit:
        if (collection_data && runGpuTask_ && enableRecHit_)
          runOnGpuRecHits(*static_cast<EcalRecHitCollection const*>(collection_data), collection);
        return runGpuTask_;
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
