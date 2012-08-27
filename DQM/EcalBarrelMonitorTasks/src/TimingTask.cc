#include "../interface/TimingTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  TimingTask::TimingTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "TimingTask"),
    energyThresholdEB_(_workerParams.getUntrackedParameter<double>("energyThresholdEB")),
    energyThresholdEE_(_workerParams.getUntrackedParameter<double>("energyThresholdEE"))
  {
    collectionMask_ = 
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit);
  }

  TimingTask::~TimingTask()
  {
  }

  bool
  TimingTask::filterRunType(const std::vector<short>& _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if ( _runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
           _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) return true;
    }

    return false;
  }

  void 
  TimingTask::runOnRecHits(const EcalRecHitCollection &_hits, Collections _collection)
  {
    uint32_t mask(~((0x1 << EcalRecHit::kGood) | (0x1 << EcalRecHit::kOutOfTime)));
    float threshold(_collection == kEBRecHit ? energyThresholdEB_ : energyThresholdEE_);

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){

      if(hitItr->checkFlagMask(mask)) continue;

      DetId id(hitItr->id());

      float time(hitItr->time());
      float energy(hitItr->energy());

      MEs_[kTimeAmp]->fill(id, energy, time);
      MEs_[kTimeAmpAll]->fill(id, energy, time);

      if(energy > threshold){
	MEs_[kTimeAll]->fill(id, time);
	MEs_[kTimeMap]->fill(id, time);
	MEs_[kTimeAllMap]->fill(id, time);
      }
    }
  }

  /*static*/
  void
  TimingTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["TimeMap"] = kTimeMap;
    _nameToIndex["TimeAmp"] = kTimeAmp;
    _nameToIndex["TimeAll"] = kTimeAll;
    _nameToIndex["TimeAllMap"] = kTimeAllMap;
    _nameToIndex["TimeAmpAll"] = kTimeAmpAll;
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}
