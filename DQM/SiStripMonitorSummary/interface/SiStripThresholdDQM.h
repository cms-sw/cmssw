#ifndef SiStripMonitorSummary_SiStripThresholdDQM_h
#define SiStripMonitorSummary_SiStripThresholdDQM_h


#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"


class SiStripThresholdDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripThresholdDQM(const edm::EventSetup & eSetup,
                      edm::ParameterSet const& hPSet,
                      edm::ParameterSet const& fPSet);
  
  ~SiStripThresholdDQM() override;
  
  void getActiveDetIds(const edm::EventSetup & eSetup) override;

   void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es) override; 
   void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es) override; 
 	       
  void fillMEsForDet(const ModMEs& selModME_,uint32_t selDetId_, const TrackerTopology* tTopo) override;
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology* tTopo) override;
  
  unsigned long long getCache(const edm::EventSetup & eSetup) override{ return eSetup.get<SiStripThresholdRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup) override{
    eSetup.get<SiStripThresholdRcd>().get(thresholdHandle_);
    cacheID_memory = cacheID_current;
  }

  private:
    edm::ESHandle<SiStripThreshold> thresholdHandle_;
    std::string WhichThreshold;
};

#endif
