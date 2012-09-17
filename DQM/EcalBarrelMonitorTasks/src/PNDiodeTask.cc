#include "../interface/PNDiodeTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  PNDiodeTask::PNDiodeTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PNDiodeTask")
  {
    collectionMask_ = 
      (0x1 << kMEMTowerIdErrors) |
      (0x1 << kMEMBlockSizeErrors) |
      (0x1 << kMEMChIdErrors) |
      (0x1 << kMEMGainErrors) |
      (0x1 << kPnDiodeDigi) |
      (0x1 << kRun);

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
    MESets set(nMESets);

    switch(_collection){
    case kMEMTowerIdErrors:
      set = kMEMTowerId;
      break;
    case kMEMBlockSizeErrors:
      set = kMEMBlockSize;
      break;
    case kMEMChIdErrors:
      set = kMEMChId;
      break;
    case kMEMGainErrors:
      set = kMEMGain;
      break;
    default:
      return;
    }

    for(EcalElectronicsIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      EcalElectronicsId eid(idItr->dccId(), idItr->towerId(), 1, idItr->xtalId());
      if(MEs_[set]) MEs_[set]->fill(eid);
    }
  }

  void
  PNDiodeTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const EcalPnDiodeDetId& id(digiItr->id());

      if(!enable_[dccId(id) - 1]) continue;

      MEs_[kOccupancy]->fill(id);
      MEs_[kOccupancySummary]->fill(id);

      float mean(0.);
      bool gainSwitch(false);

      for(int iSample(0); iSample < 4; iSample++){
	if(digiItr->sample(iSample).gainId() != 1){
	  gainSwitch = true;
	  break;
	}
	mean += digiItr->sample(iSample).adc();
      }
      if(gainSwitch) continue;

      mean /= 4.;

      MEs_[kPedestal]->fill(id, mean);
    }
  }

  /*static*/
  void
  PNDiodeTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["MEMChId"] = kMEMChId;
    _nameToIndex["MEMGain"] = kMEMGain;
    _nameToIndex["MEMBlockSize"] = kMEMBlockSize;
    _nameToIndex["MEMTowerId"] = kMEMTowerId;
    _nameToIndex["Pedestal"] = kPedestal;
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["OccupancySummary"] = kOccupancySummary;
  }

  DEFINE_ECALDQM_WORKER(PNDiodeTask);
}


