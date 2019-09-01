#ifndef SiStripMonitorSummary_SiStripCablingDQM_h
#define SiStripMonitorSummary_SiStripCablingDQM_h

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"
#include "FWCore/Framework/interface/ESHandle.h"

class SiStripCablingDQM : public SiStripBaseCondObjDQM {
public:
  SiStripCablingDQM(const edm::EventSetup &eSetup,
                    edm::RunNumber_t iRun,
                    edm::ParameterSet const &hPSet,
                    edm::ParameterSet const &fPSet);

  ~SiStripCablingDQM() override;

  void fillModMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override { ; }
  void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) override { ; }

  void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_, const TrackerTopology *tTopo) override { ; }
  void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology *tTopo) override {
    ;
  }

  void getActiveDetIds(const edm::EventSetup &eSetup) override;
  unsigned long long getCache(const edm::EventSetup &eSetup) override {
    return eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  }

  void getConditionObject(const edm::EventSetup &eSetup) override {
    eSetup.get<SiStripDetCablingRcd>().get(cablingHandle_);
    cacheID_memory = cacheID_current;
  }

private:
  //  SiStripDetInfoFileReader* reader;
  //  std::pair<std::string,uint32_t> getLayerNameAndId(const uint32_t&);
  edm::ESHandle<SiStripDetCabling> cablingHandle_;
};

#endif
