#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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

  void readModuleAndHistoList(	MonitorUserInterface			* mui,
                              	xgi::Output				* out,
			      	bool					  coll_flag);
  void plotSingleModuleHistos(	MonitorUserInterface			* mui,
                              	std::multimap<std::string, std::string> & req_map);
  void plotHistosFromPath(      MonitorUserInterface                    * mui,
                                std::multimap<std::string, std::string> & req_map);  
  void plotTkMapHistos(       	MonitorUserInterface			* mui,
                              	std::multimap<std::string, std::string> & req_map, 
			      	std::string				  sName);
  void plotTkMapHisto(       	MonitorUserInterface			* mui,
                              	std::string                               theModI, 
			      	std::string				  theMEName);
  void readModuleHistoTree(   	MonitorUserInterface			* mui, 
                              	std::string				& str_name, 
			      	xgi::Output				* out, 
			      	bool					  coll_flag);
  void readSummaryHistoTree(  	MonitorUserInterface			* mui, 
                              	std::string				& str_name, 
			      	xgi::Output				* out, 
			      	bool					  coll_flag);
  void readAlarmTree(         	MonitorUserInterface			* mui, 
                              	std::string				& str_name, 
                              	xgi::Output				* out, 
			      	bool					coll_flag);
  void plotSingleHistogram(   	MonitorUserInterface			* mui,
                              	std::multimap<std::string, std::string> & req_map);
  void readStatusMessage(     	MonitorUserInterface			* mui, 
                              	std::string				& path,
			      	xgi::Output				* out);
  void createModuleTree(      	MonitorUserInterface			* mui);
  void computeStatus(           MonitorElement                          * mui,
                                double                                  & colorValue,
				std::pair<double,double>                & norm) ;
  void getNormalization(        MonitorElement                          * mui,
                                std::pair<double,double>                & norm,
				QString                                   theMEType) ;
  void getNormalization2D(      MonitorElement                          * mui,
                                std::pair<double,double>                & normX,
                                std::pair<double,double>                & normY,
				QString                                   theMEType) ;
  void sendTkUpdatedStatus(     MonitorUserInterface			* mui,
                              	xgi::Output                             * out,
				std::string                             & meName,
				std::string                             & theTKType) ;
  void selectMEList(            MonitorUserInterface                    * mui,  
                                std::string                             & name, 
				std::vector<MonitorElement*>            & mes);
  void getMEList(               MonitorUserInterface                    * mui,  
				std::map<std::string, int>              & mEHash);
  int getDetId(                 MonitorElement                          * mui) ;				
  const std::ostringstream& getImage(                                     void)        const;
  const std::ostringstream& getIMGCImage(MonitorUserInterface		* mui,
  				std::string				  theFullPath,
				std::string				  canvasW,
				std::string				  canvasH);
  const std::ostringstream& getNamedImage( std::string                    theName);
  std::string getMEType(        MonitorElement                          * mui) ;
  

 private:

  void fillBarrelList(        	MonitorUserInterface			* mui, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  void fillEndcapList(        	MonitorUserInterface			* mui, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  void fillModuleAndHistoList(	MonitorUserInterface			* mui,
                              	std::vector<std::string>		& modules, 
			      	std::vector<std::string>		& histos);
  void selectSingleModuleHistos(MonitorUserInterface                    * mui,  
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
  void printModuleHistoList(    MonitorUserInterface 			* mui, 
                                std::ostringstream                      & str_val);
  void printSummaryHistoList(   MonitorUserInterface 			* mui, 
                                std::ostringstream                      & str_val);
  void printAlarmList(          MonitorUserInterface 			* mui, 
                                std::ostringstream                      & str_val);
  void selectImage(		std::string				& name, 
                                int                                      status);
  void selectImage(		std::string				& name, 
                                dqm::qtests::QR_map                     & test_map);
  bool goToDir(                 MonitorUserInterface                    * mui, 
                                std::string                             & sname, 
				bool                                      flg);
  bool hasItem(                 std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  std::string getItemValue(     std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  MonitorElement* getModuleME(  MonitorUserInterface                    * mui, 
                                std::string                               me_name);
  void setCanvasMessage(        const std::string                       & error_string);
  
  
  
  std::ostringstream                     pictureBuffer_ ;
  map<std::string, std::string>          namedPictureBuffer ;

  SiPixelConfigParser   	       * configParser_  ;
  SiPixelConfigWriter   	       * configWriter_  ;
  SiPixelActionExecutor 	       * actionExecutor_;
  
  TCanvas                              * theCanvas ;
  TCanvas                              * canvas_ ;
};
#endif
