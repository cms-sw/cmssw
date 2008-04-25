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
  
  void fillModMEs();
  void fillSummaryMEs();
  
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_);

  unsigned long long getCache(const edm::EventSetup & eSetup_){ return eSetup_.get<SiStripApvGainRcd>().cacheIdentifier();}

  private:
 
 };

#endif
