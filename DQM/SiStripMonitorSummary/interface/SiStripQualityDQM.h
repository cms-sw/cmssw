#ifndef SiStripMonitorSummary_SiStripQualityDQM_h
#define SiStripMonitorSummary_SiStripQualityDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

class SiStripQualityDQM : public SiStripBaseCondObjDQMGet<SiStripQuality, SiStripQualityRcd> {
public:
  SiStripQualityDQM(edm::ESGetToken<SiStripQuality, SiStripQualityRcd> token,
                    edm::RunNumber_t iRun,
                    edm::ParameterSet const &hPSet,
                    edm::ParameterSet const &fPSet,
                    const TrackerTopology *tTopo,
                    const TkDetMap *tkDetMap);

  ~SiStripQualityDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override;

  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;
  void fillGrandSummaryMEs();

private:
  int NTkBadComponent[4];  // k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];
  std::stringstream ssV[4][19];
  void SetBadComponents(int i, int component, SiStripQuality::BadComponent &BC);

  std::vector<uint32_t> alreadyFilledLayers;
};

#endif
