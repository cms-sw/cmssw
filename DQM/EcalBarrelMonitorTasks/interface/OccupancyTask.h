#ifndef OccupancyTask_H
#define OccupancyTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class OccupancyTask : public DQWorkerTask {
  public:
    OccupancyTask(const edm::ParameterSet &, const edm::ParameterSet&);
    ~OccupancyTask();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection &);
    void runOnTPDigis(const EcalTrigPrimDigiCollection &);
    void runOnRecHits(const EcalRecHitCollection &, Collections);

    enum MESets {
      kDigi, // h2f
      kDigiProjEta, // h1f
      kDigiProjPhi, // h1f
      kDigiAll,
      kDigiDCC,
      //      kRecHit, // h2f
      //      kRecHitProjEta, // h1f
      //      kRecHitProjPhi, // h1f
      kRecHit1D,
      kRecHitThr, // h2f
      kRecHitThrProjEta, // h1f
      kRecHitThrProjPhi, // h1f
      kRecHitThrAll, // h1f
      kTPDigi, // h2f
      kTPDigiProjEta, // h1f
      kTPDigiProjPhi, // h1f
      kTPDigiThr, // h2f
      kTPDigiThrProjEta, // h1f
      kTPDigiThrProjPhi, // h1f
      kTPDigiThrAll,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    float recHitThreshold_;
    float tpThreshold_;

  };

  inline void OccupancyTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    case kTrigPrimDigi:
      runOnTPDigis(*static_cast<const EcalTrigPrimDigiCollection*>(_p));
      break;
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

