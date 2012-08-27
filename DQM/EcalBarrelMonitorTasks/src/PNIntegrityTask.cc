#include "../interface/PNIntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  PNIntegrityTask::PNIntegrityTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PNIntegrityTask")
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
  PNIntegrityTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["MEMChId"] = kMEMChId;
    _nameToIndex["MEMGain"] = kMEMGain;
    _nameToIndex["MEMBlockSize"] = kMEMBlockSize;
    _nameToIndex["MEMTowerId"] = kMEMTowerId;
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityTask);
}


