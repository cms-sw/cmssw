#ifndef SiStripMonitorSummary_SiStripThresholdDQM_h
#define SiStripMonitorSummary_SiStripThresholdDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"

class SiStripThresholdDQM : public SiStripBaseCondObjDQMGet<SiStripThreshold, SiStripThresholdRcd> {
public:
  SiStripThresholdDQM(edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> token,
                      edm::RunNumber_t iRun,
                      edm::ParameterSet const &hPSet,
                      edm::ParameterSet const &fPSet,
                      const TrackerTopology *tTopo,
                      const TkDetMap *tkDetMap);

  ~SiStripThresholdDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;

private:
  std::string WhichThreshold;
};

#endif
