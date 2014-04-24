#include "../interface/TimingTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  TimingTask::TimingTask() :
    DQWorkerTask(),
    energyThresholdEB_(0.),
    energyThresholdEE_(0.)
  {
  }

  void
  TimingTask::setParams(edm::ParameterSet const& _params)
  {
    energyThresholdEB_ = _params.getUntrackedParameter<double>("energyThresholdEB");
    energyThresholdEE_ = _params.getUntrackedParameter<double>("energyThresholdEE");
  }

  bool
  TimingTask::filterRunType(short const* _runType)
  {
    for(int iFED(0); iFED < nDCC; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
         _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL) return true;
    }

    return false;
  }

  void 
  TimingTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection)
  {
    MESet& meTimeAmp(MEs_.at("TimeAmp"));
    MESet& meTimeAmpAll(MEs_.at("TimeAmpAll"));
    MESet& meTimeAll(MEs_.at("TimeAll"));
    MESet& meTimeAllMap(MEs_.at("TimeAllMap"));
    MESet& meTimeMap(MEs_.at("TimeMap"));
    MESet& meTime1D(MEs_.at("Time1D"));

    uint32_t mask(~((0x1 << EcalRecHit::kGood) | (0x1 << EcalRecHit::kOutOfTime)));
    float threshold(_collection == kEBRecHit ? energyThresholdEB_ : energyThresholdEE_);

    std::for_each(_hits.begin(), _hits.end(), [&](EcalRecHitCollection::value_type const& hit){
                    if(hit.checkFlagMask(mask)) return;

                    DetId id(hit.id());

                    float time(hit.time());
                    float energy(hit.energy());

                    meTimeAmp.fill(id, energy, time);
                    meTimeAmpAll.fill(id, energy, time);

                    if(energy > threshold){
                      meTimeAll.fill(id, time);
                      meTimeMap.fill(id, time);
                      meTime1D.fill(id, time);
                      meTimeAllMap.fill(id, time);
                    }
                  });
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}
