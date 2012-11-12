#include "../interface/PNDiodeTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  PNDiodeTask::PNDiodeTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PNDiodeTask")
  {
    collectionMask_[kMEMTowerIdErrors] = true;
    collectionMask_[kMEMBlockSizeErrors] = true;
    collectionMask_[kMEMChIdErrors] = true;
    collectionMask_[kMEMGainErrors] = true;
    collectionMask_[kPnDiodeDigi] = true;
    collectionMask_[kRun] = true;

    for(unsigned iD(0); iD < BinService::nDCC; ++iD)
      enable_[iD] = false;
  }

  bool
  PNDiodeTask::filterRunType(const std::vector<short>& _runType)
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
  PNDiodeTask::runOnErrors(const EcalElectronicsIdCollection &_ids, Collections _collection)
  {
    if(_ids.size() == 0) return;

    MESet* set(0);

    switch(_collection){
    case kMEMTowerIdErrors:
      set = MEs_["MEMTowerId"];
      break;
    case kMEMBlockSizeErrors:
      set = MEs_["MEMBlockSize"];
      break;
    case kMEMChIdErrors:
      set = MEs_["MEMChId"];
      break;
    case kMEMGainErrors:
      set = MEs_["MEMGain"];
      break;
    default:
      return;
    }

    for(EcalElectronicsIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      EcalElectronicsId eid(idItr->dccId(), idItr->towerId(), 1, idItr->xtalId());
      set->fill(eid);
    }
  }

  void
  PNDiodeTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    MESet* meOccupancy(MEs_["Occupancy"]);
    MESet* meOccupancySummary(MEs_["OccupancySummary"]);
    MESet* mePedestal(MEs_["Pedestal"]);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const EcalPnDiodeDetId& id(digiItr->id());

      if(!enable_[dccId(id) - 1]) continue;

      meOccupancy->fill(id);
      meOccupancySummary->fill(id);

      for(int iSample(0); iSample < 4; iSample++){
	if(digiItr->sample(iSample).gainId() != 1) break;
        mePedestal->fill(id, double(digiItr->sample(iSample).adc()));
      }
    }
  }

  DEFINE_ECALDQM_WORKER(PNDiodeTask);
}


