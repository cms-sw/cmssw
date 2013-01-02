#ifndef SiStripMonitorSummary_SiStripApvGainsDQM_h
#define SiStripMonitorSummary_SiStripApvGainsDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"


class SiStripApvGainsDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripApvGainsDQM(const edm::EventSetup & eSetup,
                     edm::ParameterSet const& hPSet,
                     edm::ParameterSet const& fPSet);
  
  virtual ~SiStripApvGainsDQM();

  void getActiveDetIds(const edm::EventSetup & eSetup);

  void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es);
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es);
  
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo);
  
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo);
  
  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripApvGainRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripApvGainRcd>().get(gainHandle_);
    cacheID_memory = cacheID_current;
  }
   
  private:
    edm::ESHandle<SiStripApvGain> gainHandle_;

 };

#endif
