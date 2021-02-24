#ifndef SiStripMonitorSummary_SiStripCablingDQM_h
#define SiStripMonitorSummary_SiStripCablingDQM_h

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

class SiStripCablingDQM : public SiStripBaseCondObjDQMGet<SiStripDetCabling, SiStripDetCablingRcd> {
public:
  SiStripCablingDQM(edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> token,
                    edm::RunNumber_t iRun,
                    edm::ParameterSet const &hPSet,
                    edm::ParameterSet const &fPSet,
                    const TrackerTopology *tTopo,
                    const TkDetMap *tkDetMap);

  ~SiStripCablingDQM() override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override { ; }
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override { ; }

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override { ; }
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override {
    ;
  }

  void getActiveDetIds(const edm::EventSetup &eSetup) override;
};

#endif
