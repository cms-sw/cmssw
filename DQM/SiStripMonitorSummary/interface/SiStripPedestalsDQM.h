#ifndef SiStripMonitorSummary_SiStripPedestalsDQM_h
#define SiStripMonitorSummary_SiStripPedestalsDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

class SiStripPedestalsDQM : public SiStripBaseCondObjDQM {
public:
  SiStripPedestalsDQM(const edm::EventSetup &eSetup,
                      edm::RunNumber_t iRun,
                      edm::ParameterSet const &hPSet,
                      edm::ParameterSet const &fPSet);

  ~SiStripPedestalsDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override;
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_, const TrackerTopology *tTopo) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology *tTopo) override;

  unsigned long long getCache(const edm::EventSetup &eSetup) override {
    return eSetup.get<SiStripPedestalsRcd>().cacheIdentifier();
  }

  void getConditionObject(const edm::EventSetup &eSetup) override {
    eSetup.get<SiStripPedestalsRcd>().get(pedestalHandle_);
    cacheID_memory = cacheID_current;
  }

private:
  edm::ESHandle<SiStripPedestals> pedestalHandle_;
};

#endif
