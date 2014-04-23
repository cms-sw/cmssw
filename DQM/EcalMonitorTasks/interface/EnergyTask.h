#ifndef EnergyTask_H
#define EnergyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class EnergyTask : public DQWorkerTask {
  public:
    EnergyTask();
    ~EnergyTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    void runOnRecHits(EcalRecHitCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    bool isPhysicsRun_;
    //    float threshS9_;
  };

  inline bool EnergyTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      if(_p) runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }

    return false;
  }

}

#endif

