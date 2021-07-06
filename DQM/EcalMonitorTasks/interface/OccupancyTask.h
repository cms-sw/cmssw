#ifndef OccupancyTask_H
#define OccupancyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace ecaldqm {
  class OccupancyTask : public DQWorkerTask {
  public:
    OccupancyTask();
    ~OccupancyTask() override {}

    bool filterRunType(short const*) override;

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    bool analyze(void const*, Collections) override;

    void runOnRawData(EcalRawDataCollection const&);
    template <typename DigiCollection>
    void runOnDigis(DigiCollection const&, Collections);
    void runOnTPDigis(EcalTrigPrimDigiCollection const&);
    void runOnRecHits(EcalRecHitCollection const&, Collections);
    void setEventTime(const edm::TimeValue_t& iTime);
    void setTokens(edm::ConsumesCollector&) override;

  private:
    void setParams(edm::ParameterSet const&) override;
    edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> lasertoken_;
    bool FillLaser = false;
    float recHitThreshold_;
    float tpThreshold_;
    edm::TimeValue_t m_iTime;
  };

  inline bool OccupancyTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEcalRawData:
        if (_p)
          runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
        return true;
      case kEBDigi:
        if (_p)
          runOnDigis(*static_cast<EBDigiCollection const*>(_p), _collection);
        return true;
        break;
      case kEEDigi:
        if (_p)
          runOnDigis(*static_cast<EEDigiCollection const*>(_p), _collection);
        return true;
        break;
      case kTrigPrimDigi:
        if (_p)
          runOnTPDigis(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
        return true;
        break;
      case kEBRecHit:
      case kEERecHit:
        if (_p)
          runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return true;
        break;
      default:
        break;
    }

    return false;
  }
}  // namespace ecaldqm

#endif
