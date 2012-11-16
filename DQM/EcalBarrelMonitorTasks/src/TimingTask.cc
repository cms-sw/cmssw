#include "../interface/TimingTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  TimingTask::TimingTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "TimingTask"),
    energyThresholdEB_(0.),
    energyThresholdEE_(0.)
  {
    collectionMask_ = 
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit);

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    energyThresholdEB_ = taskParams.getUntrackedParameter<double>("energyThresholdEB");
    energyThresholdEE_ = taskParams.getUntrackedParameter<double>("energyThresholdEE");
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
  TimingTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis, axisE, axisT;

    axis.low = -20.;
    axis.high = 20.;
    _data[kTimeMap] = MEData("TimeMap", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &axis);

    axis.nbins = 100;
    axis.low = -25.;
    axis.high = 25.;
    _data[kTimeAll] = MEData("TimeAll", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

    axis.low = -7.;
    axis.high = 7.;
    _data[kTimeAllMap] = MEData("TimeAllMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &axis);


    axisE.nbins = 25;
    axisE.low = -0.5;
    axisE.high = 2.;
    axisE.edges = new double[axisE.nbins + 1];
    for(int i = 0; i <= axisE.nbins; i++)
      axisE.edges[i] = pow((float)10., axisE.low + (axisE.high - axisE.low) / axisE.nbins * i);

    axisT.nbins = 200;
    axisT.low = -50.;
    axisT.high = 50.;
    axisT.edges = new double[axisT.nbins + 1];
    for(int i = 0; i <= axisT.nbins; i++)
      axisT.edges[i] = axisT.low + (axisT.high - axisT.low) / axisT.nbins * i;

    _data[kTimeAmp] = MEData("TimeAmp", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH2F, &axisE, &axisT);
    _data[kTimeAmpAll] = MEData("TimeAmpAll", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH2F, &axisE, &axisT);
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}


