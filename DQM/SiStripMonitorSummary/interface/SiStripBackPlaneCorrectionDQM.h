#ifndef SiStripMonitorSummary_SiStripBackPlaneCorrectionDQM_h
#define SiStripMonitorSummary_SiStripBackPlaneCorrectionDQM_h


#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"


class SiStripBackPlaneCorrectionDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripBackPlaneCorrectionDQM(const edm::EventSetup & eSetup,
                             edm::ParameterSet const& hPSet,
                             edm::ParameterSet const& fPSet);
  
  ~SiStripBackPlaneCorrectionDQM() override;
  
  void getActiveDetIds(const edm::EventSetup & eSetup) override;
  
  void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es) override{};
  void fillMEsForDet(const ModMEs& selModME_,uint32_t selDetId_, const TrackerTopology* tTopo) override{};
  
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es) override;
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology* tTopo) override;
  
  unsigned long long getCache(const edm::EventSetup & eSetup) override{ return eSetup.get<SiStripBackPlaneCorrectionRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup) override{
    eSetup.get<SiStripBackPlaneCorrectionRcd>().get(bpcorrectionHandle_);
    cacheID_memory = cacheID_current;
  }

  private:
    edm::ESHandle<SiStripBackPlaneCorrection> bpcorrectionHandle_;
};

#endif
