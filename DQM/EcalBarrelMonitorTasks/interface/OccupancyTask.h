#ifndef OccupancyTask_H
#define OccupancyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class OccupancyTask : public DQWorkerTask {
  public:
    OccupancyTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~OccupancyTask();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection &, Collections);
    void runOnTPDigis(const EcalTrigPrimDigiCollection &);
    void runOnRecHits(const EcalRecHitCollection &, Collections);

    enum MESets {
      kDigi, // h2f
      kDigiProjEta, // h1f
      kDigiProjPhi, // h1f
      kDigiAll,
      kDigiDCC,
      kDigi1D,
      kRecHitAll,
      kRecHitProjEta,
      kRecHitProjPhi,
      kRecHitThrProjEta, // h1f
      kRecHitThrProjPhi, // h1f
      kRecHitThrAll, // h1f
      kRecHitThr1D,
      kTPDigiProjEta, // h1f
      kTPDigiProjPhi, // h1f
      kTPDigiAll, // h2f
      kTPDigiThrProjEta, // h1f
      kTPDigiThrProjPhi, // h1f
      kTPDigiThrAll,
      kTrendNDigi,
      kTrendNRecHitThr,
      kTrendNTPDigi,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  private:
    float recHitThreshold_;
    float tpThreshold_;

  };

  inline void OccupancyTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p), _collection);
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

