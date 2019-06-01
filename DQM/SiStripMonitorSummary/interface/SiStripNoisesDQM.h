#ifndef SiStripMonitorSummary_SiStripNoisesDQM_h
#define SiStripMonitorSummary_SiStripNoisesDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

class SiStripNoisesDQM : public SiStripBaseCondObjDQM {
public:
  SiStripNoisesDQM(const edm::EventSetup &eSetup,
                   edm::RunNumber_t iRun,
                   edm::ParameterSet const &hPSet,
                   edm::ParameterSet const &fPSet);

  ~SiStripNoisesDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_, const TrackerTopology *tTopo) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology *tTopo) override;

  unsigned long long getCache(const edm::EventSetup &eSetup) override {
    return eSetup.get<SiStripNoisesRcd>().cacheIdentifier();
  }

  void getConditionObject(const edm::EventSetup &eSetup) override {
    eSetup.get<SiStripNoisesRcd>().get(noiseHandle_);
    cacheID_memory = cacheID_current;
  }

private:
  bool gainRenormalisation_;
  bool simGainRenormalisation_;
  edm::ESHandle<SiStripNoises> noiseHandle_;
  edm::ESHandle<SiStripApvGain> gainHandle_;
};

#endif
