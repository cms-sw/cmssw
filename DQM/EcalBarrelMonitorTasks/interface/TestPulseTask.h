#ifndef TestPulseTask_H
#define TestPulseTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class TestPulseTask : public DQWorkerTask {
  public:
    TestPulseTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~TestPulseTask();

    void bookMEs() override;

    bool filterRunType(const std::vector<short>&) override;

    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void endEvent(const edm::Event&, const edm::EventSetup&) override;

    void analyze(const void*, Collections) override;

    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&);

    enum Constants {
      nGain = 3,
      nPNGain = 2
    };

    enum MESets{
      kOccupancy,
      kShape = kOccupancy + nGain,
      kAmplitude = kShape + nGain, // profile2d
      kPNOccupancy = kAmplitude + nGain, // profile2d
      kPNAmplitude = kPNOccupancy + nPNGain, // profile2d
      nMESets = kPNAmplitude + nPNGain
    };

    static void setMEData(std::vector<MEData>&);

  protected:
    bool enable_[54];
    int gain_[54];
    std::vector<int> MGPAGains_;
    std::vector<int> MGPAGainsPN_;
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
    case kEBUncalibRecHit:
    case kEEUncalibRecHit:
      runOnUncalibRecHits(*static_cast<const EcalUncalibratedRecHitCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif
