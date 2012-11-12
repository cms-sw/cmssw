#ifndef EnergyTask_H
#define EnergyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//class CaloTopology;

namespace ecaldqm {

  class EnergyTask : public DQWorkerTask {
  public:
    EnergyTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~EnergyTask();

    bool filterRunType(const std::vector<short>&);

/*     void beginRun(const edm::Run &, const edm::EventSetup &); */

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &);

  private:
    //    const CaloTopology *topology_;
    bool isPhysicsRun_;
    //    float threshS9_;
  };

  inline void EnergyTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      runOnRecHits(*static_cast<const EcalRecHitCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif

