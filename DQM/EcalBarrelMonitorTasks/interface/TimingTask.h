#ifndef TimingTask_H
#define TimingTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TimingTask : public DQWorkerTask {
  public:
    TimingTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TimingTask();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &, Collections);

  private:
    float energyThresholdEB_;
    float energyThresholdEE_;
  };

  inline void TimingTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      runOnRecHits(*static_cast<const EcalRecHitCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

