#ifndef LaserTask_H
#define LaserTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LaserTask : public DQWorkerTask {
  public:
    LaserTask();
    ~LaserTask() override {}

    void addDependencies(DependencySet&) override;

    bool filterRunType(short const*) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;

    bool analyze(void const*, Collections) override;

    void runOnRawData(EcalRawDataCollection const&);
    template <typename DigiCollection>
    void runOnDigis(DigiCollection const&);
    void runOnPnDigis(EcalPnDiodeDigiCollection const&);
    void runOnUncalibRecHits(EcalUncalibratedRecHitCollection const&);

    enum Wavelength { kGreen, kBlue, kIRed, nWavelength };

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> wlToME_;

    bool enable_[nDCC];
    unsigned wavelength_[nDCC];
    unsigned rtHalf_[nDCC];
    std::map<uint32_t, float> pnAmp_;

    int emptyLS_;
    int emptyLSLimit_;
    int maxPedestal_;
  };

  inline bool LaserTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEcalRawData:
        if (_p)
          runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
        return true;
        break;
      case kEBDigi:
        if (_p)
          runOnDigis(*static_cast<EBDigiCollection const*>(_p));
        return true;
        break;
      case kEEDigi:
        if (_p)
          runOnDigis(*static_cast<EEDigiCollection const*>(_p));
        return true;
        break;
      case kPnDiodeDigi:
        if (_p)
          runOnPnDigis(*static_cast<EcalPnDiodeDigiCollection const*>(_p));
        return true;
        break;
      case kEBLaserLedUncalibRecHit:
      case kEELaserLedUncalibRecHit:
        if (_p)
          runOnUncalibRecHits(*static_cast<EcalUncalibratedRecHitCollection const*>(_p));
        return true;
        break;
      default:
        break;
    }

    return false;
  }
}  // namespace ecaldqm

#endif
