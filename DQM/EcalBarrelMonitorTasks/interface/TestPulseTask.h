#ifndef TestPulseTask_H
#define TestPulseTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TestPulseTask : public DQWorkerTask {
  public:
    TestPulseTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TestPulseTask() {}

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&);

    enum MESets{
      kOccupancy,
      kShape,
      kAmplitude,
      kPNAmplitude,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    bool enable_[BinService::nDCC];
    int gain_[BinService::nDCC];
  };

  inline void TestPulseTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    case kEBTestPulseUncalibRecHit:
    case kEETestPulseUncalibRecHit:
      runOnUncalibRecHits(*static_cast<const EcalUncalibratedRecHitCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif
