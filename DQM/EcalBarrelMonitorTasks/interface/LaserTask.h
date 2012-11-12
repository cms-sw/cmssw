#ifndef LaserTask_H
#define LaserTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LaserTask : public DQWorkerTask {
  public:
    LaserTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LaserTask() {}

    void setDependencies(DependencySet&);

    bool filterRunType(const std::vector<short>&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRawData(EcalRawDataCollection const&);
    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);
    void runOnUncalibRecHits(const EcalUncalibratedRecHitCollection&);

  private:
    std::map<int, unsigned> wlToME_;

    bool enable_[BinService::nDCC];
    unsigned wavelength_[BinService::nDCC];
    unsigned rtHalf_[BinService::nDCC];
    std::map<uint32_t, float> pnAmp_;

    int emptyLS_;
    int emptyLSLimit_;
  };

  inline void LaserTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEcalRawData:
      runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
      break;
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    case kEBLaserLedUncalibRecHit:
    case kEELaserLedUncalibRecHit:
      runOnUncalibRecHits(*static_cast<const EcalUncalibratedRecHitCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif
