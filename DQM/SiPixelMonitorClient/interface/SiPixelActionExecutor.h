#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

// For Tracker Map
enum funcType {EachBinContent, Entries, Mean, Sum, WeightedSum};
#define PI_12 0.261799
#define PI    3.141592
#define PI_2  1.570796

#define NLev1		4		// Number of HalfCylinders in Endcap or number of Shells in Barrel, which is bigger
#define NLev2		3		// Number of Disks in Endcap or number of Layers in Barrel, which is bigger
#define NLev3		22		// Number of Blades in Endcap or number of Ladders in Barrel, which is bigger
#define NLev4		7		// Number of Modules - different for Endcap and Barrel, which is bigger

#define NCyl		4
#define NDisk		2
#define NBlade		12
#define NModuleE	7

#define NShell		4
#define NLayer		3
//#define NLadders	LayNum * 6 + 4		// where LayNum is number of interesting Layer => 10, 16, 22
#define NModuleB	4

#define NPoints		5

// End for Tracker Map

class SiPixelActionExecutor {

 public:

  SiPixelActionExecutor(            bool                           offlineXMLfile,
                                    bool                           Tier0Flag);
 ~SiPixelActionExecutor();

 void createSummary(    	    DQMStore    		 * bei,
 				    bool   			   isUpgrade);
 void bookDeviations(               DQMStore                     * bei,
 				    bool			   isUpgrade);
 void bookEfficiency(    	    DQMStore    		 * bei,
 				    bool			   isUpgrade);
 void createEfficiency(    	    DQMStore    		 * bei,
 				    bool			   isUpgrade);
 void fillEfficiency(    	    DQMStore    		 * bei,
                                    bool                           isbarrel,
				    bool			   isUpgrade);
 void bookOccupancyPlots(    	    DQMStore    		 * bei,
                                    bool                           hiRes,
									bool				isbarrel);
 void bookOccupancyPlots(    	    DQMStore    		 * bei,
                                    bool                           hiRes);
 void createOccupancy(    	    DQMStore    		 * bei);
 void setupQTests(      	    DQMStore    		 * bei);
 void checkQTestResults(	    DQMStore    		 * bei);
 void createTkMap(      	    DQMStore    		 * bei, 
                        	    std::string 	    	   mEName,
				    std::string 	    	   theTKType);
 bool readConfiguration(	    int 			 & tkmap_freq, 
                        	    int 			 & sum_barrel_freq, 
				    int 			 & sum_endcap_freq, 
				    int 			 & sum_grandbarrel_freq, 
				    int 			 & sum_grandendcap_freq,
				    int 			 & message_limit,
                                    int                          & source_type,
				    int                          & calib_type);
 bool readConfiguration(	    int 			 & tkmap_freq, 
                        	    int 			 & summary_freq);
 void readConfiguration(	    );
 void createLayout(     	    DQMStore    		 * bei);
 void fillLayout(       	    DQMStore    		 * bei);
 int getTkMapMENames(               std::vector<std::string>	 & names);
 void dumpModIds(                   DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpBarrelModIds(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpEndcapModIds(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpRefValues(                   DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpBarrelRefValues(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpEndcapRefValues(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void createMaps(DQMStore* bei, std::string type, std::string name, funcType ff);
 void bookTrackerMaps(DQMStore* bei, std::string name);


private:
  
  
  MonitorElement* getSummaryME(     DQMStore     		 * bei, 
                                    std::string 	     	   me_name,
				    bool			   isUpgrade);
  MonitorElement* getFEDSummaryME(  DQMStore     		 * bei, 
                                    std::string 	     	   me_name);
  void GetBladeSubdirs(DQMStore* bei, std::vector<std::string>& blade_subdirs); 
  void fillSummary(           	    DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names,
				    bool isbarrel,
				    bool isUpgrade);
  void fillDeviations(              DQMStore     		 * bei);
  void fillFEDErrorSummary(         DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillGrandBarrelSummaryHistos(DQMStore     		 * bei, 
			            std::vector<std::string> 	 & me_names,
				    bool 			   isUpgrade);
  void fillGrandEndcapSummaryHistos(DQMStore     		 * bei, 
			            std::vector<std::string> 	 & me_names,
				    bool			   isUpgrade);
  void getGrandSummaryME(           DQMStore     		 * bei,
                                    int                      	   nbin, 
                                    std::string              	 & me_name, 
				    std::vector<MonitorElement*> & mes);
 
  void fillOccupancy(    	    DQMStore    		 * bei,
				    bool isbarrel);

  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  int message_limit_;
  int source_type_;
  int calib_type_;  
  int ndet_;
  bool offlineXMLfile_;
  bool Tier0Flag_;
  
  QTestHandle* qtHandler_;
  
  MonitorElement * OccupancyMap;
  MonitorElement * PixelOccupancyMap;
  MonitorElement * HitEfficiency_L1;
  MonitorElement * HitEfficiency_L2;
  MonitorElement * HitEfficiency_L3;
  MonitorElement * HitEfficiency_L4;
  MonitorElement * HitEfficiency_Dp1;
  MonitorElement * HitEfficiency_Dp2;
  MonitorElement * HitEfficiency_Dp3;
  MonitorElement * HitEfficiency_Dm1;
  MonitorElement * HitEfficiency_Dm2;
  MonitorElement * HitEfficiency_Dm3;
  MonitorElement * DEV_adc_Barrel;
  MonitorElement * DEV_ndigis_Barrel;
  MonitorElement * DEV_charge_Barrel;
  MonitorElement * DEV_nclusters_Barrel;
  MonitorElement * DEV_size_Barrel;
  MonitorElement * DEV_adc_Endcap;
  MonitorElement * DEV_ndigis_Endcap;
  MonitorElement * DEV_charge_Endcap;
  MonitorElement * DEV_nclusters_Endcap;
  MonitorElement * DEV_size_Endcap;
  
  
  int createMap(Double_t map[][NLev2][NLev3][NLev4], std::string type, DQMStore* bei, funcType ff, bool isBarrel);
  void getData(Double_t map[][NLev2][NLev3][NLev4], std::string type, DQMStore* bei, funcType ff, Int_t i, Int_t j, Int_t k, Int_t l);
  void prephistosB(MonitorElement* me[NCyl], DQMStore *bei, const Double_t map[][NLev2][NLev3][NLev4], std::string name, Double_t min, Double_t max);
  void prephistosE(MonitorElement* me[NCyl], DQMStore *bei, const Double_t map[][NLev2][NLev3][NLev4], std::string name, Double_t min, Double_t max);
  Double_t mapMax(const Double_t map[][NLev2][NLev3][NLev4], bool isBarrel); 
  Double_t mapMin(const Double_t map[][NLev2][NLev3][NLev4], bool isBarrel);

  TH2F * temp_H;
  TH2F * temp_1x2;
  TH2F * temp_1x5;
  TH2F * temp_2x3;
  TH2F * temp_2x4;
  TH2F * temp_2x5;
  
};
#endif
