#ifndef SiStripMonitorSummary_SiStripCablingDQM_h
#define SiStripMonitorSummary_SiStripCablingDQM_h


#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

class SiStripCablingDQM: public SiStripBaseCondObjDQM{
  
  public:

  SiStripCablingDQM(const edm::EventSetup & eSetup,
		    edm::ParameterSet const& hPSet,
		    edm::ParameterSet const& fPSet);
  
  ~SiStripCablingDQM();

  void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){;}
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){;}

  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){;}
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){;}

  void getActiveDetIds(const edm::EventSetup & eSetup);
  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripDetCablingRcd>().get(cablingHandle_);
    cacheID_memory = cacheID_current;
  }

 
  private:
  
  //  SiStripDetInfoFileReader* reader; 
  //  std::pair<std::string,uint32_t> getLayerNameAndId(const uint32_t&);
  edm::ESHandle<SiStripDetCabling> cablingHandle_;  
};

#endif
