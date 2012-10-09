#ifndef EnergyTask_H
#define EnergyTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class CaloTopology;

namespace ecaldqm {

  class EnergyTask : public DQWorkerTask {
  public:
    EnergyTask(const edm::ParameterSet &, const edm::ParameterSet&);
    ~EnergyTask();

    bool filterRunType(const std::vector<short>&);

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &);

    enum MESets {
      kHitMap, // profile2d
      kHitMapAll,
      //      k3x3Map, // profile2d 
      kHit, // h1f
      kHitAll,
      kMiniCluster, // h1f
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    const CaloTopology *topology_;
    bool isPhysicsRun_;
    float threshS9_;
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

