#include "../interface/PNIntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  PNIntegrityTask::PNIntegrityTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "PNIntegrityTask")
  {
    collectionMask_ = 
      (0x1 << kMEMTowerIdErrors) |
      (0x1 << kMEMBlockSizeErrors) |
      (0x1 << kMEMChIdErrors) |
      (0x1 << kMEMGainErrors);
  }

  PNIntegrityTask::~PNIntegrityTask()
  {
  }

  void
  PNIntegrityTask::runOnErrors(const EcalElectronicsIdCollection &_ids, Collections _collection)
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
      if(MEs_[set]) MEs_[set]->fill(*idItr);
    }
  }

  /*static*/
  void
  PNIntegrityTask::setMEData(std::vector<MEData>& _data)
  {
    _data[kMEMChId] = MEData("MEMChId", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kMEMGain] = MEData("MEMGain", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kMEMBlockSize] = MEData("MEMBlockSize", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kMEMTowerId] = MEData("MEMTowerId", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityTask);
}


