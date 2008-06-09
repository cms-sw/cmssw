#ifndef SiStripMonitorSummary_SiStripQualityDQM_h
#define SiStripMonitorSummary_SiStripQualityDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"


class SiStripQualityDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripQualityDQM(const edm::EventSetup & eSetup_,
                         edm::ParameterSet const& hPSet,
                         edm::ParameterSet const& fPSet);
  
  virtual ~SiStripQualityDQM();
  
  void fillModMEs();
  void fillSummaryMEs();
  
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_);
 
  unsigned long long getCache(const edm::EventSetup & eSetup_){ return eSetup_.get<SiStripQualityRcd>().cacheIdentifier();}

  private:
    std::string qualityLabel_ ;
};

#endif
