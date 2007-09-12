#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"


#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "TCanvas.h"
#include "qstring.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <map>

class SiPixelInformationExtractor {

 public:

  SiPixelInformationExtractor();
 ~SiPixelInformationExtractor();

  //void readModuleAndHistoList(	MonitorUserInterface			* mui,
  void readModuleAndHistoList(	DaqMonitorBEInterface			* bei,
                              	xgi::Output				* out,
			      	bool					  coll_flag);
  //void plotSingleModuleHistos(	MonitorUserInterface			* mui,
  void plotSingleModuleHistos(	DaqMonitorBEInterface			* bei,
                              	std::multimap<std::string, std::string> & req_map);
  //void plotHistosFromPath(      MonitorUserInterface                    * mui,
  void plotHistosFromPath(      DaqMonitorBEInterface                   * bei,
                                std::multimap<std::string, std::string> & req_map);  
  //void plotTkMapHistos(       	MonitorUserInterface			* mui,
  void plotTkMapHistos(       	DaqMonitorBEInterface			* bei,
                              	std::multimap<std::string, std::string> & req_map, 
			      	std::string				  sName);
  //void plotTkMapHisto(       	MonitorUserInterface			* mui,
  void plotTkMapHisto(       	DaqMonitorBEInterface			* bei,
                              	std::string                               theModI, 
			      	std::string				  theMEName);
  //void readModuleHistoTree(   	MonitorUserInterface			* mui, 
  void readModuleHistoTree(   	DaqMonitorBEInterface			* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out, 
			      	bool					  coll_flag);
  //void readSummaryHistoTree(  	MonitorUserInterface			* mui, 
  void readSummaryHistoTree(  	DaqMonitorBEInterface			* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out, 
			      	bool					  coll_flag);
  //void readAlarmTree(         	MonitorUserInterface			* mui, 
  void readAlarmTree(         	DaqMonitorBEInterface			* bei, 
                              	std::string				& str_name, 
                              	xgi::Output				* out, 
			      	bool					coll_flag);
  //void plotSingleHistogram(   	MonitorUserInterface			* mui,
  void plotSingleHistogram(   	DaqMonitorBEInterface			* bei,
                              	std::multimap<std::string, std::string> & req_map);
  //void readStatusMessage(     	MonitorUserInterface			* mui, 
  void readStatusMessage(     	DaqMonitorBEInterface			* bei, 
                              	std::string				& path,
			      	xgi::Output				* out);
  //void createModuleTree(      	MonitorUserInterface			* mui);
  void createModuleTree(      	DaqMonitorBEInterface			* bei);
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
  //void sendTkUpdatedStatus(     MonitorUserInterface			* mui,
  void sendTkUpdatedStatus(     DaqMonitorBEInterface			* bei,
                              	xgi::Output                             * out,
				std::string                             & meName,
				std::string                             & theTKType) ;
  //void selectMEList(            MonitorUserInterface                    * mui,  
  void selectMEList(            DaqMonitorBEInterface                   * bei,  
                                std::string                             & name, 
				std::vector<MonitorElement*>            & mes);
  //void getMEList(               MonitorUserInterface                    * mui,  
  void getMEList(               DaqMonitorBEInterface                   * bei,  
				std::map<std::string, int>              & mEHash);
  int getDetId(                 MonitorElement                          * mE) ;				
  const std::ostringstream& getImage(                                     void)        const;
  //const std::ostringstream& getIMGCImage(MonitorUserInterface		* mui,
  const std::ostringstream& getIMGCImage(DaqMonitorBEInterface		* bei,
  				std::string				  theFullPath,
				std::string				  canvasW,
				std::string				  canvasH);
  const std::ostringstream& getNamedImage( std::string                    theName);
  std::string getMEType(        MonitorElement                          * mE) ;
  

 private:

  //void fillBarrelList(        	MonitorUserInterface			* mui, 
  void fillBarrelList(        	DaqMonitorBEInterface			* bei, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  //void fillEndcapList(        	MonitorUserInterface			* mui, 
  void fillEndcapList(        	DaqMonitorBEInterface			* bei, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  //void fillModuleAndHistoList(	MonitorUserInterface			* mui,
  void fillModuleAndHistoList(	DaqMonitorBEInterface			* bei,
                              	std::vector<std::string>		& modules, 
			      	std::map<std::string,std::string>	& histos);
  //void selectSingleModuleHistos(MonitorUserInterface                    * mui,  
  void selectSingleModuleHistos(DaqMonitorBEInterface                   * bei,  
                                std::string                               mid, 
                                std::vector<std::string>                & names, 
				std::vector<MonitorElement*>            & mes);
  void getItemList(             std::multimap<std::string, std::string> & req_map,
                                std::string                               item_name, 
				std::vector<std::string>                & items);
  void fillImageBuffer();
  void fillImageBuffer(         TCanvas                                 & c1);
  void fillNamedImageBuffer(    TCanvas                                 * c1,
                                std::string                               theName);
  void plotHistos(              std::multimap<std::string, std::string> & req_map, 
                                std::vector<MonitorElement*>              me_list);
  void plotHisto(               MonitorElement                          * theMe,
                                std::string                               theName,
				std::string 				  canvasW,
				std::string 				  canvasH);
  //void printModuleHistoList(    MonitorUserInterface 			* mui, 
  void printModuleHistoList(    DaqMonitorBEInterface 			* bei, 
                                std::ostringstream                      & str_val);
  //void printSummaryHistoList(   MonitorUserInterface 			* mui, 
  void printSummaryHistoList(   DaqMonitorBEInterface 			* bei, 
                                std::ostringstream                      & str_val);
  //void printAlarmList(          MonitorUserInterface 			* mui, 
  void printAlarmList(          DaqMonitorBEInterface 			* bei, 
                                std::ostringstream                      & str_val);
  void selectImage(		std::string				& name, 
                                int                                      status);
  void selectImage(		std::string				& name, 
                                dqm::qtests::QR_map                     & test_map);
  //bool goToDir(                 MonitorUserInterface                    * mui, 
  bool goToDir(                 DaqMonitorBEInterface                   * bei, 
                                std::string                             & sname, 
				bool                                      flg);
  bool hasItem(                 std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  std::string getItemValue(     std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  //MonitorElement* getModuleME(  MonitorUserInterface                    * mui, 
  MonitorElement* getModuleME(  DaqMonitorBEInterface                   * bei, 
                                std::string                               me_name);
  void setCanvasMessage(        const std::string                       & error_string);
  
  
  
  std::ostringstream                     pictureBuffer_ ;
  map<std::string, std::string>          namedPictureBuffer ;
  
  int                                    alarmCounter_;

  SiPixelConfigParser   	       * configParser_  ;
  SiPixelConfigWriter   	       * configWriter_  ;
  SiPixelActionExecutor 	       * actionExecutor_;
  
  TCanvas                              * theCanvas ;
  TCanvas                              * canvas_ ;
};
#endif
