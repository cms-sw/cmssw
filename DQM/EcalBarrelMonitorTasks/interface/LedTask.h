#ifndef LedTask_H
#define LedTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LedTask : public DQWorkerTask {
  public:
    LedTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LedTask() {}

    bool filterRunType(const std::vector<short>&);
    bool filterEventSetting(const std::vector<EventSettings>&);

    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&);

    enum MESets {
      kAmplitudeSummary,
      kAmplitude,
      kOccupancy,
      kShape,
      kTiming,
      kAOverP,
      kPNAmplitude,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  private:
    std::map<int, unsigned> wlToME_;

    bool enable_[BinService::nDCC];
    int wavelength_[BinService::nDCC];
    std::map<int, std::vector<float> > pnAmp_;
  };

  inline void LedTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    case kEELaserLedUncalibRecHit:
      runOnUncalibRecHits(*static_cast<const EcalUncalibratedRecHitCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif
