#ifndef TimingTask_H
#define TimingTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TimingTask : public DQWorkerTask {
  public:
    TimingTask();
    ~TimingTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    void runOnRecHits(EcalRecHitCollection const&, Collections);
    void runOnUncalibRecHits(EcalUncalibratedRecHitCollection const&);

  private:
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void setParams(edm::ParameterSet const&) override;

    float chi2ThresholdEB_;
    float chi2ThresholdEE_;
    float energyThresholdEB_;
    float energyThresholdEE_;

    MESet* meTimeMapByLS;
  };

  inline bool TimingTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      if(_p) runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
      return true;
      break;
    case kEBUncalibRecHit:
    case kEEUncalibRecHit:
      if(_p) runOnUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif

