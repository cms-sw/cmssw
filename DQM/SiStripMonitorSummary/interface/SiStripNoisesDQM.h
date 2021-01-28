#ifndef SiStripMonitorSummary_SiStripNoisesDQM_h
#define SiStripMonitorSummary_SiStripNoisesDQM_h

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

class SiStripNoisesDQM : public SiStripBaseCondObjDQMGet<SiStripNoises, SiStripNoisesRcd> {
public:
  SiStripNoisesDQM(edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken,
                   edm::RunNumber_t iRun,
                   edm::ParameterSet const& hPSet,
                   edm::ParameterSet const& fPSet,
                   const TrackerTopology* tTopo,
                   const TkDetMap* tkDetMap,
                   const SiStripApvGain* gainHandle);

  ~SiStripNoisesDQM() override;

  void getActiveDetIds(const edm::EventSetup& eSetup) override;

  void fillMEsForDet(const ModMEs& selModME_, uint32_t selDetId_) override;
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_) override;

private:
  const SiStripApvGain* gainHandle_ = nullptr;
};

#endif
