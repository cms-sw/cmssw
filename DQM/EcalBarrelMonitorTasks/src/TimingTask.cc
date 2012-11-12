#include "../interface/TimingTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  TimingTask::TimingTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "TimingTask"),
    energyThresholdEB_(_workerParams.getUntrackedParameter<double>("energyThresholdEB")),
    energyThresholdEE_(_workerParams.getUntrackedParameter<double>("energyThresholdEE"))
  {
    collectionMask_[kEBRecHit] = true;
    collectionMask_[kEERecHit] = true;
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
    MESet* meTimeAmp(MEs_["TimeAmp"]);
    MESet* meTimeAmpAll(MEs_["TimeAmpAll"]);
    MESet* meTimeAll(MEs_["TimeAll"]);
    MESet* meTimeAllMap(MEs_["TimeAllMap"]);
    MESet* meTimeMap(MEs_["TimeMap"]);
    MESet* meTime1D(MEs_["Time1D"]);

    uint32_t mask(~((0x1 << EcalRecHit::kGood) | (0x1 << EcalRecHit::kOutOfTime)));
    float threshold(_collection == kEBRecHit ? energyThresholdEB_ : energyThresholdEE_);

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){

      if(hitItr->checkFlagMask(mask)) continue;

      DetId id(hitItr->id());

      float time(hitItr->time());
      float energy(hitItr->energy());

      meTimeAmp->fill(id, energy, time);
      meTimeAmpAll->fill(id, energy, time);

      if(energy > threshold){
	meTimeAll->fill(id, time);
	meTimeMap->fill(id, time);
        meTime1D->fill(id, time);
	meTimeAllMap->fill(id, time);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}
