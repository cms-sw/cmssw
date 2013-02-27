#ifndef LedTask_H
#define LedTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LedTask : public DQWorkerTask {
  public:
    LedTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~LedTask();

    bool filterRunType(const std::vector<short>&);

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void endEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRawData(const EcalRawDataCollection&);
    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&);

    enum Constants {
      nWL = 2,
      nPNGain = 2
    };

    enum MESets {
      kAmplitudeSummary, // profile2d
      kAmplitude = kAmplitudeSummary + nWL, // profile2d
      kOccupancy = kAmplitude + nWL,
      kShape = kOccupancy + nWL,
      kTiming = kShape + nWL, // profile2d
      kAOverP = kTiming + nWL, // profile2d
      kPNAmplitude = kAOverP + nWL, // profile2d
      kPNOccupancy = kPNAmplitude + nWL * nPNGain, // profile2d
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    std::vector<int> ledWavelengths_;
    std::vector<int> MGPAGainsPN_;

    bool enable_[BinService::nDCC];
    int wavelength_[BinService::nDCC];
    std::map<int, std::vector<float> > pnAmp_;
  };

  inline void LedTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEcalRawData:
      runOnRawData(*static_cast<const EcalRawDataCollection*>(_p));
      break;
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
