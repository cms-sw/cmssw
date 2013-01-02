#ifndef SiStripMonitorSummary_SiStripQualityDQM_h
#define SiStripMonitorSummary_SiStripQualityDQM_h


#include "FWCore/Framework/interface/ESHandle.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"

/* #include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h" */

/* #include "CondFormats/SiStripObjects/interface/SiStripNoises.h" */
/* #include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h" */
/* #include "CondFormats/SiStripObjects/interface/SiStripPedestals.h" */
/* #include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h" */

/* #include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h" */
/* #include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h" */

/* #include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h" */


/* #include "CalibTracker/Records/interface/SiStripDetCablingRcd.h" */
/* #include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h" */

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"



class SiStripQualityDQM : public SiStripBaseCondObjDQM{
 
  public:
  
  SiStripQualityDQM(const edm::EventSetup & eSetup,
                      edm::ParameterSet const& hPSet,
                      edm::ParameterSet const& fPSet);
  
  virtual ~SiStripQualityDQM();
  
  void getActiveDetIds(const edm::EventSetup & eSetup);

  void fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es);
  void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo);
  
  void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es);
  void fillMEsForLayer( /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo);
  void fillGrandSummaryMEs(const edm::EventSetup& eSetup);
 	       
  
  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripQualityRcd>().cacheIdentifier();}
  
  void getConditionObject(const edm::EventSetup & eSetup){
    eSetup.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle_);
    cacheID_memory = cacheID_current;
  }

  private: 
    std::string qualityLabel_ ;
    edm::ESHandle<SiStripQuality> qualityHandle_;
    int NTkBadComponent[4]; //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
    int NBadComponent[4][19][4];  
    std::stringstream ssV[4][19];
    void SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC);

    std::vector<uint32_t> alreadyFilledLayers;
};

#endif
