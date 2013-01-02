#ifndef SiStripMonitorSummary_SiStripLorentzAngleDQM_h
#define SiStripMonitorSummary_SiStripLorentzAngleDQM_h


#include "FWCore/Framework/interface/ESHandle.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"


class SiStripLorentzAngleDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripLorentzAngleDQM(const edm::EventSetup & eSetup,
                             edm::ParameterSet const& hPSet,
                             edm::ParameterSet const& fPSet);
  
  virtual ~SiStripLorentzAngleDQM();
  
  void getActiveDetIds(const edm::EventSetup & eSetup);
  
  void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){};
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){};
  
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es);
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo);
  
  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripLorentzAngleRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripLorentzAngleRcd>().get(lorentzangleHandle_);
    cacheID_memory = cacheID_current;
  }

  private:
    edm::ESHandle<SiStripLorentzAngle> lorentzangleHandle_;
};

#endif
