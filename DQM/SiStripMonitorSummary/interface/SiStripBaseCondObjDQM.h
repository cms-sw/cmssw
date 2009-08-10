#ifndef SiStripMonitorSummary_SiStripBaseCondObjDQM_h
#define SiStripMonitorSummary_SiStripBaseCondObjDQM_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"  /// ADDITON OF TK_HISTO_MAP
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <vector>
#include <map>
#include <string>


class SiStripBaseCondObjDQM {

 

  public:
  
		 

  SiStripBaseCondObjDQM(const edm::EventSetup & eSetup,
			edm::ParameterSet const& hPSet,
			edm::ParameterSet const& fPSet );
  
  virtual ~SiStripBaseCondObjDQM(){};
  
  virtual void getActiveDetIds(const edm::EventSetup & eSetup)=0;

    void analysis(const edm::EventSetup & eSetup_);
    void analysisOnDemand(const edm::EventSetup & eSetup_, uint32_t detIdOnDemand);
    void analysisOnDemand(const edm::EventSetup & eSetup_, std::vector<uint32_t> detIdsOnDemand);
    void analysisOnDemand(const edm::EventSetup & eSetup_,
                        std::string requestedSubDetector, 
                        uint32_t requestedSide, 
			uint32_t requestedLayer);
  

    std::vector<uint32_t> getCabledModules();
    void selectModules(std::vector<uint32_t> & detIds_);
  
    //    virtual void fillTopSummaryMEs()=0;
 
    virtual unsigned long long getCache(const edm::EventSetup & eSetup_)=0;
    virtual void getConditionObject(const edm::EventSetup & eSetup_)=0;

    virtual void end();
      
  protected:
    
    struct ModMEs{ 
      MonitorElement* ProfileDistr;
      MonitorElement* CumulDistr;
      MonitorElement* SummaryOfProfileDistr;
      MonitorElement* SummaryOfCumulDistr;
      MonitorElement* SummaryDistr;
    }; 


    void getModMEs(ModMEs& CondObj_ME, const uint32_t& detId_);
    void getSummaryMEs(ModMEs& CondObj_ME, const uint32_t& detId_);
    std::pair<std::string,uint32_t> getLayerNameAndId(const uint32_t& detId_);
    std::pair<std::string,uint32_t> getStringNameAndId(const uint32_t& detId_);
    std::vector<uint32_t> GetSameLayerDetId(std::vector<uint32_t> activeDetIds, uint32_t selDetId);


    virtual void fillModMEs(const std::vector<uint32_t> & selectedDetIds);
    virtual void fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds);
    virtual void fillMEsForDet(ModMEs selModME_,uint32_t selDetId_)=0;
    virtual void fillMEsForLayer( std::map<uint32_t, ModMEs> selModMEsMap_, uint32_t selDetId_)=0;


    void fillTkMap(const uint32_t& detid, const float& value);
    
    SiStripDetInfoFileReader* reader; 
    
    const edm::EventSetup& eSetup_;
    edm::ParameterSet hPSet_;
    edm::ParameterSet fPSet_;
    
    bool Mod_On_;
    bool HistoMaps_On_;
    bool SummaryOnLayerLevel_On_;
    bool SummaryOnStringLevel_On_;
    bool GrandSummary_On_;
    double minValue, maxValue;
    std::vector<int> tkMapScaler;

    //bool ActiveDetIds_On_;

    std::string CondObj_fillId_;
    std::string CondObj_name_;

    std::map<uint32_t, ModMEs> ModMEsMap_;
    std::map<uint32_t, ModMEs> SummaryMEsMap_;
    std::vector<uint32_t> activeDetIds;
    std::vector<uint32_t> all_DetIds;
 
    unsigned long long cacheID_memory;
    unsigned long long cacheID_current;

    TkHistoMap* Tk_HM_;
    TkHistoMap* Tk_HM_H;
    TkHistoMap* Tk_HM_L;
    TrackerMap * tkMap;
  
 private:
  
    void bookProfileMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_);
    void bookCumulMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_);
    void bookSummaryProfileMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_);
    void bookSummaryCumulMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_);
    void bookSummaryMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_);
    void bookTkMap(const std::string& TkMapname);

    void saveTkMap(const std::string& TkMapname, double minValue, double maxValue);

    std::vector<uint32_t> ModulesToBeExcluded_;
    std::vector<uint32_t> ModulesToBeIncluded_;
    std::vector<std::string> SubDetectorsToBeExcluded_;

    edm::ESHandle<SiStripDetCabling> detCablingHandle_;

    std::string condDataMonitoringMode_;
     
    SiStripHistoId hidmanager;                        
    SiStripFolderOrganizer folder_organizer;         
    DQMStore* dqmStore_;

};


#endif
