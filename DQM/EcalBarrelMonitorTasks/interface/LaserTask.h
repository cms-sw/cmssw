#ifndef LaserTask_H
#define LaserTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LaserTask : public DQWorkerTask {
  public:
    LaserTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~LaserTask();

    bool filterRunType(const std::vector<short>&);

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void endEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRawData(const EcalRawDataCollection&);
    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&, Collections);

    std::vector<int> const& getLaserWavelengths() const { return laserWavelengths_; }
    std::vector<int> const& getMGPAGainsPN() const { return MGPAGainsPN_; }

    enum Constants {
      nWL = 4,
      nPNGain = 2
    };

    enum MESets {
      kAmplitudeSummary, // profile2d
      kAmplitude = kAmplitudeSummary + nWL, // profile2d
      kOccupancy = kAmplitude + nWL,
      kTiming = kOccupancy + nWL, // profile2d
      kShape = kTiming + nWL,
      kAOverP = kShape + nWL, // profile2d
      kPNAmplitude = kAOverP + nWL, // profile2d
      kPNOccupancy = kPNAmplitude + nWL * nPNGain, // profile2d
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    std::vector<int> laserWavelengths_;
    std::vector<int> MGPAGainsPN_;

    bool enable_[BinService::nDCC];
    int wavelength_[BinService::nDCC];
    std::map<int, std::vector<float> > pnAmp_;
  };

  inline void LaserTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEcalRawData:
      runOnRawData(*static_cast<const EcalRawDataCollection*>(_p));
      break;
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    case kEBUncalibRecHit:
    case kEEUncalibRecHit:
      runOnUncalibRecHits(*static_cast<const EcalUncalibratedRecHitCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif
