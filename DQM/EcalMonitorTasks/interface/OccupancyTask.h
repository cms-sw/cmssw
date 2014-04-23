#ifndef OccupancyTask_H
#define OccupancyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm
{
  class OccupancyTask : public DQWorkerTask {
  public:
    OccupancyTask();
    ~OccupancyTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    void runOnRawData(EcalRawDataCollection const&);
    template<typename DigiCollection> void runOnDigis(DigiCollection const&, Collections);
    void runOnTPDigis(EcalTrigPrimDigiCollection const&);
    void runOnRecHits(EcalRecHitCollection const&, Collections);

  private:
    void setParams(edm::ParameterSet const&) override;

    float recHitThreshold_;
    float tpThreshold_;
  };

  inline bool OccupancyTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEcalRawData:
      if(_p) runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
      return true;
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p), _collection);
      return true;
      break;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p), _collection);
      return true;
      break;
    case kTrigPrimDigi:
      if(_p) runOnTPDigis(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
      return true;
      break;
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

