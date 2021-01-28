#ifndef SiStripMonitorSummary_SiStripLorentzAngleDQM_h
#define SiStripMonitorSummary_SiStripLorentzAngleDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

class SiStripLorentzAngleDQM : public SiStripBaseCondObjDQMGet<SiStripLorentzAngle, SiStripLorentzAngleRcd> {
public:
  SiStripLorentzAngleDQM(edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> token,
                         edm::RunNumber_t iRun,
                         edm::ParameterSet const &hPSet,
                         edm::ParameterSet const &fPSet,
                         const TrackerTopology *tTopo,
                         const TkDetMap *tkDetMap);

  ~SiStripLorentzAngleDQM() override;

  void getActiveDetIds(const edm::EventSetup &eSetup) override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds) override{};
  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_) override{};

  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;
};

#endif
