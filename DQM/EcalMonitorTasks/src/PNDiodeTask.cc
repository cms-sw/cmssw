#include "DQM/EcalMonitorTasks/interface/PNDiodeTask.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace ecaldqm {

  PNDiodeTask::PNDiodeTask() : DQWorkerTask() { std::fill_n(enable_, nDCC, false); }

  bool PNDiodeTask::filterRunType(short const* _runType) {
    bool enable(false);

    for (int iDCC(0); iDCC < 54; iDCC++) {
      if (_runType[iDCC] == EcalDCCHeaderBlock::LASER_STD || _runType[iDCC] == EcalDCCHeaderBlock::LASER_GAP ||
          _runType[iDCC] == EcalDCCHeaderBlock::LED_STD || _runType[iDCC] == EcalDCCHeaderBlock::LED_GAP ||
          _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_MGPA || _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_GAP ||
          _runType[iDCC] == EcalDCCHeaderBlock::PEDESTAL_STD || _runType[iDCC] == EcalDCCHeaderBlock::PEDESTAL_GAP) {
        enable = true;
        enable_[iDCC] = true;
      } else
        enable_[iDCC] = false;
    }

    return enable;
  }

  void PNDiodeTask::runOnErrors(EcalElectronicsIdCollection const& _ids, Collections _collection) {
    if (_ids.empty())
      return;

    MESet* meMEMErrors = &MEs_.at("MEMErrors");

    // MEM Box Integrity Errors (TowerIds 69 and 70)
    // errorType matches to the following labels in DQM plot
    // 0 = TOWERID
    // 1 = BLOCKSIZE
    // 2 = CHID
    // 3 = GAIN
    int errorType(-1);
    switch (_collection) {
      case kMEMTowerIdErrors:
        errorType = 0;
        break;
      case kMEMBlockSizeErrors:
        errorType = 1;
        break;
      case kMEMChIdErrors:
        errorType = 2;
        break;
      case kMEMGainErrors:
        errorType = 3;
        break;
      default:
        return;
    }

    // Integrity errors for MEM boxes (towerIds 69/70)
    // Plot contains two bins per dccId. Integer number
    // bins correspond to towerId 69 and half integer
    // number bins correspond to towerId 70.
    std::for_each(_ids.begin(),
                  _ids.end(),
                  [&](EcalElectronicsIdCollection::value_type const& id) {
                    if (id.towerId() == 69)
                      meMEMErrors->fill(id.dccId() + 0.0, errorType);
                    else if (id.towerId() == 70)
                      meMEMErrors->fill(id.dccId() + 0.5, errorType);
                    else {
                      edm::LogWarning("EcalDQM")
                          << "PNDiodeTask::runOnErrors : one of the ids in the electronics ID collection does not "
                          << "correspond to one of the MEM box towerIds (69/70) in lumi number " << timestamp_.iLumi
                          << ", event number " << timestamp_.iEvt;
                    }
                  });
  }

  void PNDiodeTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis) {
    MESet& meOccupancy(MEs_.at("Occupancy"));
    MESet& meOccupancySummary(MEs_.at("OccupancySummary"));
    MESet& mePedestal(MEs_.at("Pedestal"));

    std::for_each(_digis.begin(), _digis.end(), [&](EcalPnDiodeDigiCollection::value_type const& digi) {
      const EcalPnDiodeDetId& id(digi.id());

      if (!enable_[dccId(id) - 1])
        return;

      meOccupancy.fill(id);
      meOccupancySummary.fill(id);

      for (int iSample(0); iSample < 4; iSample++) {
        if (digi.sample(iSample).gainId() != 1)
          break;
        mePedestal.fill(id, double(digi.sample(iSample).adc()));
      }
    });
  }

  DEFINE_ECALDQM_WORKER(PNDiodeTask);
}  // namespace ecaldqm
