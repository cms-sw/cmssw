#ifndef SiStripMonitorSummary_SiStripPedestalsDQM_h
#define SiStripMonitorSummary_SiStripPedestalsDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

class SiStripPedestalsDQM : public SiStripBaseCondObjDQMGet<SiStripPedestals, SiStripPedestalsRcd> {
public:
  SiStripPedestalsDQM(edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> token,
                      edm::RunNumber_t iRun,
                      edm::ParameterSet const &hPSet,
                      edm::ParameterSet const &fPSet,
                      const TrackerTopology *tTopo,
                      const TkDetMap *tkDetMap);

  ~SiStripPedestalsDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;
};

#endif
