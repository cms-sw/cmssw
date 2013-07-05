#ifndef TimingTask_H
#define TimingTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TimingTask : public DQWorkerTask {
  public:
    TimingTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~TimingTask();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &, Collections);

    enum MESets {
      kTimeMap,
      kTimeAmp,
      kTimeAll,
      kTimeAllMap,
      kTimeAmpAll,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

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

