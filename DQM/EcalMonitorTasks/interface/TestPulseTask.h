#ifndef TestPulseTask_H
#define TestPulseTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TestPulseTask : public DQWorkerTask {
  public:
    TestPulseTask();
    ~TestPulseTask() {}

    bool filterRunType(short const*) override;

    void addDependencies(DependencySet&) override;

    bool analyze(void const*, Collections) override;

    void runOnRawData(EcalRawDataCollection const&);
    template<typename DigiCollection> void runOnDigis(DigiCollection const&);
    void runOnPnDigis(EcalPnDiodeDigiCollection const&);
    void runOnUncalibRecHits(EcalUncalibratedRecHitCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    bool enable_[nDCC];
    int gain_[nDCC];
  };

  inline bool TestPulseTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEcalRawData:
      if(_p) runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
      return true;
      break;
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p));
      return true;
      break;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p));
      return true;
      break;
    case kPnDiodeDigi:
      if(_p) runOnPnDigis(*static_cast<EcalPnDiodeDigiCollection const*>(_p));
      return true;
      break;
    case kEBTestPulseUncalibRecHit:
    case kEETestPulseUncalibRecHit:
      if(_p) runOnUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif
