#ifndef LedTask_H
#define LedTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace ecaldqm {

  class LedTask : public DQWorkerTask {
  public:
    LedTask();
    ~LedTask() override {}

    void addDependencies(DependencySet&) override;

    bool filterRunType(short const*) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;

    bool analyze(void const*, Collections) override;

    void runOnRawData(EcalRawDataCollection const&);
    void runOnDigis(EEDigiCollection const&);
    void runOnPnDigis(EcalPnDiodeDigiCollection const&);
    void runOnUncalibRecHits(EcalUncalibratedRecHitCollection const&);

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> wlToME_;

    bool enable_[nEEDCC];
    unsigned wavelength_[nEEDCC];
    unsigned rtHalf_[nEEDCC];
    std::map<unsigned, float> pnAmp_;

    int emptyLS_;
    int emptyLSLimit_;
    int isemptyLS;
  };

  inline bool LedTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEcalRawData:
        if (_p)
          runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
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
