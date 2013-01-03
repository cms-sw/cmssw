#ifndef SiStripMonitorSummary_SiStripNoisesDQM_h
#define SiStripMonitorSummary_SiStripNoisesDQM_h


#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"

class SiStripNoisesDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripNoisesDQM(const edm::EventSetup & eSetup,
                   edm::ParameterSet const& hPSet,
                   edm::ParameterSet const& fPSet);
  
  virtual ~SiStripNoisesDQM();
  
  void getActiveDetIds(const edm::EventSetup & eSetup);

  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_, const TrackerTopology* tTopo);
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology* tTopo);

  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripNoisesRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripNoisesRcd>().get(noiseHandle_);
    cacheID_memory = cacheID_current;
  }
 
  private:
    bool gainRenormalisation_;
    edm::ESHandle<SiStripNoises> noiseHandle_; 
    edm::ESHandle<SiStripApvGain> gainHandle_;
    
};

#endif
