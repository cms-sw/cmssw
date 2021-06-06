#ifndef SiStripMonitorSummary_SiStripBackPlaneCorrectionDQM_h
#define SiStripMonitorSummary_SiStripBackPlaneCorrectionDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

class SiStripBackPlaneCorrectionDQM
    : public SiStripBaseCondObjDQMGet<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> {
public:
  SiStripBackPlaneCorrectionDQM(edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> token,
                                edm::RunNumber_t iRun,
                                edm::ParameterSet const &hPSet,
                                edm::ParameterSet const &fPSet,
                                const TrackerTopology *tTopo,
                                const TkDetMap *tkDetMap);

  ~SiStripBackPlaneCorrectionDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override{};
  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override{};

  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;
};

#endif
