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

  private:
    void setParams(edm::ParameterSet const&) override;

    float energyThresholdEB_;
    float energyThresholdEE_;
  };

  inline bool TimingTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      if(_p) runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif

