#ifndef SiStripMonitorSummary_SiStripPedestalsDQM_h
#define SiStripMonitorSummary_SiStripPedestalsDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"


class SiStripPedestalsDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripPedestalsDQM(const edm::EventSetup & eSetup,
                      edm::ParameterSet const& hPSet,
                      edm::ParameterSet const& fPSet);
  
  virtual ~SiStripPedestalsDQM();
  
  void fillModMEs();
  void fillSummaryMEs();
 	       
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_);
  void fillMEsForLayer( std::map<uint32_t, ModMEs> selModMEsMap_, uint32_t selDetId_);
  
  unsigned long long getCache(const edm::EventSetup & eSetup_){ return eSetup_.get<SiStripPedestalsRcd>().cacheIdentifier();}

  private:

};

#endif
