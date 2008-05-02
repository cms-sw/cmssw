#ifndef SiStripMonitorSummary_SiStripNoisesDQM_h
#define SiStripMonitorSummary_SiStripNoisesDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"


class SiStripNoisesDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripNoisesDQM(const edm::EventSetup & eSetup,
                   edm::ParameterSet const& hPSet,
                   edm::ParameterSet const& fPSet);
  
  virtual ~SiStripNoisesDQM();
  
  void fillModMEs();
  void fillSummaryMEs();

  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_);
  void fillMEsForLayer( std::map<uint32_t, ModMEs> selModMEsMap_, uint32_t selDetId_);

  unsigned long long getCache(const edm::EventSetup & eSetup_){ return eSetup_.get<SiStripNoisesRcd>().cacheIdentifier();}

  private:

};

#endif
