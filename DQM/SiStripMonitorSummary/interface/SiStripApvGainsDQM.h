#ifndef SiStripMonitorSummary_SiStripApvGainsDQM_h
#define SiStripMonitorSummary_SiStripApvGainsDQM_h

#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

class SiStripApvGainsDQM : public SiStripBaseCondObjDQM {
public:
  SiStripApvGainsDQM(const edm::EventSetup &eSetup,
                     edm::RunNumber_t iRun,
                     edm::ParameterSet const &hPSet,
                     edm::ParameterSet const &fPSet);

  ~SiStripApvGainsDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override;
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_, const TrackerTopology *tTopo) override;

  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology *tTopo) override;

  unsigned long long getCache(const edm::EventSetup &eSetup) override {
    return eSetup.get<SiStripApvGainRcd>().cacheIdentifier();
  }

  void getConditionObject(const edm::EventSetup &eSetup) override {
    eSetup.get<SiStripApvGainRcd>().get(gainHandle_);
    cacheID_memory = cacheID_current;
  }

private:
  edm::ESHandle<SiStripApvGain> gainHandle_;
};

#endif
