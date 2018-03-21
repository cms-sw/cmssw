#include "../interface/PNDiodeTask.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace ecaldqm {

  PNDiodeTask::PNDiodeTask() :
    DQWorkerTask()
  {
    std::fill_n(enable_, nDCC, false);
  }

  bool
  PNDiodeTask::filterRunType(short const* _runType)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < 54; iDCC++){
      if(_runType[iDCC] == EcalDCCHeaderBlock::LASER_STD ||
         _runType[iDCC] == EcalDCCHeaderBlock::LASER_GAP ||
         _runType[iDCC] == EcalDCCHeaderBlock::LED_STD ||
         _runType[iDCC] == EcalDCCHeaderBlock::LED_GAP ||
         _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
         _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_GAP ||
         _runType[iDCC] == EcalDCCHeaderBlock::PEDESTAL_STD ||
         _runType[iDCC] == EcalDCCHeaderBlock::PEDESTAL_GAP){
        enable = true;
        enable_[iDCC] = true;
      }
      else
        enable_[iDCC] = false;
    }

    return enable;
  }

  void
  PNDiodeTask::runOnErrors(EcalElectronicsIdCollection const& _ids, Collections _collection)
  {
    if(_ids.empty()) return;

    MESet* set(nullptr);

    switch(_collection){
    case kMEMTowerIdErrors:
      set = &MEs_.at("MEMTowerId");
      break;
    case kMEMBlockSizeErrors:
      set = &MEs_.at("MEMBlockSize");
      break;
    case kMEMChIdErrors:
      set = &MEs_.at("MEMChId");
      break;
    case kMEMGainErrors:
      set = &MEs_.at("MEMGain");
      break;
    default:
      return;
    }

    std::for_each(_ids.begin(), _ids.end(), [&](EcalElectronicsIdCollection::value_type const& id){
        if (id.towerId() == 69) {
          edm::LogWarning("EcalDQM") << "PNDiodeTask::runOnErrors : one of the ids in the electronics ID collection is unphysical in lumi number " << timestamp_.iLumi << ", event number " << timestamp_.iEvt; // Added March 2018 because some events have this unphysical tower ID and cause the ECAL calibration application to crash.
        }
        else {
          set->fill(EcalElectronicsId(id.dccId(), id.towerId(), 1, id.xtalId()));
        }
      });
  }

  void
  PNDiodeTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis)
  {
    MESet& meOccupancy(MEs_.at("Occupancy"));
    MESet& meOccupancySummary(MEs_.at("OccupancySummary"));
    MESet& mePedestal(MEs_.at("Pedestal"));

    std::for_each(_digis.begin(), _digis.end(), [&](EcalPnDiodeDigiCollection::value_type const& digi){
        const EcalPnDiodeDetId& id(digi.id());

        if(!enable_[dccId(id) - 1]) return;

        meOccupancy.fill(id);
        meOccupancySummary.fill(id);

        for(int iSample(0); iSample < 4; iSample++){
          if(digi.sample(iSample).gainId() != 1) break;
          mePedestal.fill(id, double(digi.sample(iSample).adc()));
        }
      });
  }

  DEFINE_ECALDQM_WORKER(PNDiodeTask);
}


