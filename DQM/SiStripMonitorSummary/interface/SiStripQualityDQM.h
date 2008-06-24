#ifndef SiStripMonitorSummary_SiStripQualityDQM_h
#define SiStripMonitorSummary_SiStripQualityDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"



class SiStripQualityDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripQualityDQM(const edm::EventSetup & eSetup,
                      edm::ParameterSet const& hPSet,
                      edm::ParameterSet const& fPSet);
  
  virtual ~SiStripQualityDQM();
  
  void getActiveDetIds(const edm::EventSetup & eSetup);

  void fillModMEs(const std::vector<uint32_t> & selectedDetIds);
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_);
  
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds);
  void fillMEsForLayer( std::map<uint32_t, ModMEs> selModMEsMap_, uint32_t selDetId_);
 	       
  
  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripQualityRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle_);
    cacheID_memory = cacheID_current;
  }

  private: 
    std::string qualityLabel_ ;
    edm::ESHandle<SiStripQuality> qualityHandle_;

};

#endif
