#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelLayoutParser.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "TCanvas.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TH2F.h"
#include "TGaxis.h"
#include "qstring.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <map>
class DQMStore;
class SiPixelEDAClient;
class SiPixelWebInterface;
class SiPixelHistoPlotter;
class SiPixelInformationExtractor {

 public:

  SiPixelInformationExtractor();
 ~SiPixelInformationExtractor();

  void getSingleModuleHistos(   DQMStore                                * bei, 
                                const std::multimap<std::string, std::string>& req_map, 
				xgi::Output                             * out);
  void getHistosFromPath(       DQMStore                                * bei, 
                                const std::multimap<std::string, std::string>& req_map, 
				xgi::Output                             * out);
  void getTrackerMapHistos(     DQMStore                                * bei, 
                                const std::multimap<std::string, std::string>& req_map, 
				xgi::Output                             * out);
				
				
  void readModuleAndHistoList(	DQMStore				* bei,
                              	xgi::Output				* out);
  void readModuleHistoTree(   	DQMStore				* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out);
  void readSummaryHistoTree(  	DQMStore				* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out);
  void readAlarmTree(         	DQMStore				* bei, 
                              	std::string				& str_name, 
                              	xgi::Output				* out);
  void readStatusMessage(       DQMStore                                * bei, 
                                std::multimap<std::string, std::string>& req_map, 
				xgi::Output * out);
  void computeStatus(           MonitorElement                          * mE,
                                double                                  & colorValue,
				std::pair<double,double>                & norm) ;
  void getNormalization(        MonitorElement                          * mE,
                                std::pair<double,double>                & norm,
				QString                                   theMEType) ;
  void getNormalization2D(      MonitorElement                          * mE,
                                std::pair<double,double>                & normX,
                                std::pair<double,double>                & normY,
				QString                                   theMEType) ;
  void sendTkUpdatedStatus(     DQMStore				* bei,
                              	xgi::Output                             * out,
				std::string                             & meName,
				std::string                             & theTKType) ;
  void selectMEList(            DQMStore                                * bei,  
                                std::string                             & name, 
				std::vector<MonitorElement*>            & mes);
  void getMEList(               DQMStore                                * bei,  
				std::map<std::string, int>              & mEHash);
  int getDetId(                 MonitorElement                          * mE) ;				
  void getIMGCImage(            const std::multimap<std::string, std::string>& req_map, 
                                xgi::Output                             * out);
  void getIMGCImage(            std::multimap<std::string, std::string>& req_map, 
                                xgi::Output                             * out);
  std::string getMEType(        MonitorElement                          * mE) ;
    
  void readConfiguration();
  bool readConfiguration(        std::map<std::string,std::vector< std::string> >   & layoutMap,
				 std::map<std::string,std::map<std::string,std::string> >                & qtestsMap,
				 std::map<std::string,std::vector<std::string> >    & meQTestsMap);

  void bookGlobalQualityFlag    (DQMStore                               * bei);

  void computeGlobalQualityFlag (DQMStore                               * bei,
                                 bool                                     init);
  
  void fillGlobalQualityPlot    (DQMStore                               * bei,
                                 bool                                     init,
                                 edm::EventSetup const                  & eSetup);
  
  void createImages             (DQMStore                               * bei);
  
 private:

  void fillModuleAndHistoList(	DQMStore				* bei,
                              	std::vector<std::string>		& modules, 
			      	std::map<std::string,std::string>	& histos);
  void getItemList(             const std::multimap<std::string, std::string> & req_map,
                                std::string                               item_name, 
				std::vector<std::string>                & items);
  void printModuleHistoList(    DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  void printSummaryHistoList(   DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  void printAlarmList(          DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  bool goToDir(                 DQMStore                                * bei, 
                                std::string                             & sname);
  bool hasItem(                 std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  std::string getItemValue(     const std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  std::string getItemValue(     std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  void createDummiesFromLayout();  
  void selectImage(            std::string                              & name, 
                               int                                        status);
  void selectImage(            std::string                              & name, 
                               std::vector<QReport*>                    & reports);
  void selectColor(            std::string                              & col, 
                               int                                        status);
  void selectColor(            std::string                              & col, 
                               std::vector<QReport*>                    & reports);
  
  void setHTMLHeader(          xgi::Output                              * out);
  void setXMLHeader(           xgi::Output                              * out);
  void setPlainHeader(         xgi::Output                              * out);
 
  int                                    alarmCounter_;

  SiPixelConfigParser   	       * configParser_  ;
  SiPixelConfigWriter   	       * configWriter_  ;
  SiPixelActionExecutor 	       * actionExecutor_;
  SiPixelLayoutParser                  * layoutParser_  ;

  std::map<std::string, 
           std::vector< std::string> >  layoutMap;
  std::map<std::string, 
           std::map<std::string, 
                    std::string> >      qtestsMap;
  std::map<std::string, 
           std::vector<std::string> >   meQTestsMap;

  
  TCanvas                              * theCanvas ;
  TCanvas                              * canvas_ ;
  TCanvas                              * canvasSlide_ ;
  TPaveText                            * paveOnCanvas;

  bool  readReference_;
  bool  readQTestMap_;
  bool  readMeMap_;
  bool  flagHotModule_;
  
  float qflag_, bpix_flag_, shellmI_flag_, shellmO_flag_, shellpI_flag_;
  float shellpO_flag_, fpix_flag_, hcylmI_flag_, hcylmO_flag_;
  float hcylpI_flag_, hcylpO_flag_;
  int allMods_, bpix_mods_, shellmI_mods_, shellmO_mods_, shellpI_mods_;
  int shellpO_mods_, fpix_mods_, hcylmI_mods_, hcylmO_mods_;
  int hcylpI_mods_, hcylpO_mods_;
  int errorMods_, err_bpix_mods_, err_shellmI_mods_, err_shellmO_mods_;
  int err_shellpI_mods_, err_shellpO_mods_, err_fpix_mods_, err_hcylmI_mods_;
  int err_hcylmO_mods_, err_hcylpI_mods_, err_hcylpO_mods_; 
  
  TH2F * allmodsEtaPhi;
  TH2F * errmodsEtaPhi;
  TH2F * goodmodsEtaPhi;
  int count;
  int errcount;
  
  MonitorElement * SummaryReport;
  MonitorElement * SummaryReportMap;
  MonitorElement * SummaryBarrel;
  MonitorElement * SummaryShellmI;
  MonitorElement * SummaryShellmO;
  MonitorElement * SummaryShellpI;
  MonitorElement * SummaryShellpO;
  MonitorElement * SummaryEndcap;
  MonitorElement * SummaryHCmI;
  MonitorElement * SummaryHCmO;
  MonitorElement * SummaryHCpI;
  MonitorElement * SummaryHCpO;
  
  SiPixelHistoPlotter* histoPlotter_;
};
#endif
