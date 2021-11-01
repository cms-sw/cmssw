#include "DQM/EcalMonitorTasks/interface/GpuTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {
  GpuTask::GpuTask() : DQWorkerTask() {}

  void GpuTask::addDependencies(DependencySet& _dependencies) {
    // Ensure we run on CpuRecHits before GpuRecHits
    _dependencies.push_back(Dependency(kEBGpuRecHit, kEBCpuRecHit));
    _dependencies.push_back(Dependency(kEEGpuRecHit, kEECpuRecHit));
  }

  bool GpuTask::filterRunType(short const* _runType) {
    for (unsigned iFED(0); iFED != ecaldqm::nDCC; iFED++) {
      if (_runType[iFED] == EcalDCCHeaderBlock::COSMIC || _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL || _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void GpuTask::beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) {
    EBCpuRecHits_ = nullptr;
    EECpuRecHits_ = nullptr;
  }

  void GpuTask::runOnCpuRecHits(EcalRecHitCollection const& _hits, Collections _collection) {
    MESet& meRecHitCpu(MEs_.at("RecHitCpu"));
    MESet& meRecHitCpuEnergy(MEs_.at("RecHitCpuEnergy"));
    MESet& meRecHitCpuTime(MEs_.at("RecHitCpuTime"));
    MESet& meRecHitCpuFlags(MEs_.at("RecHitCpuFlags"));

    int iSubdet(_collection == kEBCpuRecHit ? EcalBarrel : EcalEndcap);

    // Save CpuRecHits for comparison with GpuRecHits
    if (iSubdet == EcalBarrel)
      EBCpuRecHits_ = &_hits;
    else
      EECpuRecHits_ = &_hits;

    unsigned nCpuHits(_hits.size());
    meRecHitCpu.fill(getEcalDQMSetupObjects(), iSubdet, nCpuHits);

    for (EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr) {
      float cpuEnergy(hitItr->energy());
      if (cpuEnergy < 0.)
        continue;

      float cpuTime(hitItr->time());
      uint32_t cpuFlags(hitItr->flagsBits());

      meRecHitCpuEnergy.fill(getEcalDQMSetupObjects(), iSubdet, cpuEnergy);
      meRecHitCpuTime.fill(getEcalDQMSetupObjects(), iSubdet, cpuTime);
      meRecHitCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, cpuFlags);
    }
  }

  // Should always run after GpuTask::runOnGpuRecHits()
  void GpuTask::runOnGpuRecHits(EcalRecHitCollection const& _gpuHits, Collections _collection) {
    MESet& meRecHitGpuCpu(MEs_.at("RecHitGpuCpu"));
    MESet& meRecHitGpuCpuEnergy(MEs_.at("RecHitGpuCpuEnergy"));
    MESet& meRecHitGpuCpuTime(MEs_.at("RecHitGpuCpuTime"));
    MESet& meRecHitGpuCpuFlags(MEs_.at("RecHitGpuCpuFlags"));

    int iSubdet(_collection == kEBGpuRecHit ? EcalBarrel : EcalEndcap);

    // Get CpuRecHits saved from GpuTask::runOnCpuRecHits() for this event
    // Note: _gpuHits is a collection and cpuHits is a pointer to a collection
    EcalRecHitCollection const* cpuHits = (iSubdet == EcalBarrel) ? EBCpuRecHits_ : EECpuRecHits_;
    if (!cpuHits) {
      edm::LogWarning("EcalDQM") << "GpuTask: Did not find " << ((iSubdet == EcalBarrel) ? "EB" : "EE")
                                 << "CpuRecHits Collection. Aborting runOnGpuRecHits\n";
      return;
    }

    unsigned nGpuHits(_gpuHits.size());
    unsigned nCpuHits(cpuHits->size());
    meRecHitGpuCpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuHits - nCpuHits);

    for (EcalRecHitCollection::const_iterator gpuItr(_gpuHits.begin()); gpuItr != _gpuHits.end(); ++gpuItr) {
      // Find CpuRecHit with matching DetId
      DetId gpuId(gpuItr->detid());
      EcalRecHitCollection::const_iterator cpuItr(cpuHits->find(gpuId));
      if (cpuItr == cpuHits->end()) {
        edm::LogWarning("EcalDQM") << "GpuTask: Did not find DetId " << gpuId.rawId() << " in a CPU collection\n";
        continue;
      }

      float gpuEnergy(gpuItr->energy());
      float cpuEnergy(cpuItr->energy());

      float gpuTime(gpuItr->time());
      float cpuTime(cpuItr->time());

      uint32_t gpuFlags(gpuItr->flagsBits());
      uint32_t cpuFlags(cpuItr->flagsBits());

      meRecHitGpuCpuEnergy.fill(getEcalDQMSetupObjects(), iSubdet, gpuEnergy - cpuEnergy);
      meRecHitGpuCpuTime.fill(getEcalDQMSetupObjects(), iSubdet, gpuTime - cpuTime);
      meRecHitGpuCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, gpuFlags - cpuFlags);
    }
  }

  DEFINE_ECALDQM_WORKER(GpuTask);
}  // namespace ecaldqm
