#ifndef SiStripMonitorSummary_SiStripApvGainsDQM_h
#define SiStripMonitorSummary_SiStripApvGainsDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

class SiStripApvGainsDQM : public SiStripBaseCondObjDQMGet<SiStripApvGain, SiStripApvGainRcd> {
public:
  SiStripApvGainsDQM(edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> token,
                     edm::RunNumber_t iRun,
                     edm::ParameterSet const &hPSet,
                     edm::ParameterSet const &fPSet,
                     const TrackerTopology *tTopo,
                     const TkDetMap *tkDetMap);

  ~SiStripApvGainsDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override;

  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;
};

#endif
