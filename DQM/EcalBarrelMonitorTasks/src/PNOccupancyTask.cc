#include "../interface/PNOccupancyTask.h"

namespace ecaldqm
{
  PNOccupancyTask::PNOccupancyTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PNOccupancyTask")
  {
    collectionMask_ = (0x1 << kPnDiodeDigi);

    for(unsigned iD(0); iD < BinService::nDCC; ++iD)
      enable_[iD] = false;
  }

  bool
  PNOccupancyTask::filterRunType(std::vector<short> const& _runType)
  {
    bool enable(false);

    for(int iFED(0); iFED < BinService::nDCC; ++iFED){
      if(_runType[iFED] == EcalDCCHeaderBlock::LASER_STD ||
         _runType[iFED] == EcalDCCHeaderBlock::LASER_GAP ||
         _runType[iFED] == EcalDCCHeaderBlock::LED_STD ||
         _runType[iFED] == EcalDCCHeaderBlock::LED_GAP ||
         _runType[iFED] == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
         _runType[iFED] == EcalDCCHeaderBlock::TESTPULSE_GAP ||
         _runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_STD ||
         _runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_GAP)
        enable = true;
      enable_[iFED] = true;
    }

    return enable;
  }

  void
  PNOccupancyTask::runOnDigis(EcalPnDiodeDigiCollection const& _pnDigis)
  {
    EcalPnDiodeDigiCollection::const_iterator dEnd(_pnDigis.end());
    for(EcalPnDiodeDigiCollection::const_iterator dItr(_pnDigis.begin()); dItr != dEnd; ++dItr){
      EcalPnDiodeDetId const& id(dItr->id());

      if(!enable_[dccId(id) - 1]) continue;

      MEs_[kDigi]->fill(id);
    }
  }
  
  /*static*/
  void
  PNOccupancyTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Digi"] = kDigi;
  }

  DEFINE_ECALDQM_WORKER(PNOccupancyTask);
}
