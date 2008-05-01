#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
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
class SiPixelEDAClient;
class SiPixelWebInterface;

class SiPixelInformationExtractor {

 public:

  SiPixelInformationExtractor();
 ~SiPixelInformationExtractor();

  void readModuleAndHistoList(	DQMStore				* bei,
                              	xgi::Output				* out);
  void plotSingleModuleHistos(	DQMStore				* bei,
                              	std::multimap<std::string, std::string> & req_map);
  void plotHistosFromPath(      DQMStore               	                * bei,
                                std::multimap<std::string, std::string> & req_map);  
  void plotTkMapHistos(       	DQMStore				* bei,
                              	std::multimap<std::string, std::string> & req_map, 
			      	std::string				  sName);
  void plotTkMapHisto(       	DQMStore				* bei,
                              	std::string                               theModI, 
			      	std::string				  theMEName);
  void readModuleHistoTree(   	DQMStore				* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out);
  void readSummaryHistoTree(  	DQMStore				* bei, 
                              	std::string				& str_name, 
			      	xgi::Output				* out);
  void readAlarmTree(         	DQMStore				* bei, 
                              	std::string				& str_name, 
                              	xgi::Output				* out);
  void plotSingleHistogram(   	DQMStore				* bei,
                              	std::multimap<std::string, std::string> & req_map);
  void readStatusMessage(     	DQMStore				* bei, 
                              	std::string				& path,
			      	xgi::Output				* out);
  void createModuleTree(      	DQMStore				* bei);
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
  const std::ostringstream& getImage(                                     void)        const;
  const std::ostringstream& getIMGCImage(DQMStore			* bei,
  				std::string				  theFullPath,
				std::string				  canvasW,
				std::string				  canvasH);
  const std::ostringstream& getNamedImage( std::string                    theName);
  std::string getMEType(        MonitorElement                          * mE) ;
  
  void   readLayoutNames(       xgi::Output                             * out);
  void   plotErrorOverviewHistos(DQMStore                               * bei);
  
  void readConfiguration();
  bool readConfiguration(        std::map<std::string,std::vector< std::string> >   & layoutMap,
				 std::map<std::string,std::map<std::string,std::string> >                & qtestsMap,
				 std::map<std::string,std::vector<std::string> >    & meQTestsMap);

  void computeGlobalQualityFlag (DQMStore                               * bei,
                                 bool                                     init);
  
  void fillGlobalQualityPlot    (DQMStore                               * bei,
                                 bool                                     init,
                                 edm::EventSetup const                  & eSetup);
  
  
 private:

  void fillBarrelList(        	DQMStore				* bei, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  void fillEndcapList(        	DQMStore				* bei, 
                              	std::string				  dir_name,
                              	std::vector<std::string>		& me_names);
  void fillModuleAndHistoList(	DQMStore				* bei,
                              	std::vector<std::string>		& modules, 
			      	std::map<std::string,std::string>	& histos);
  void selectSingleModuleHistos(DQMStore                                * bei,  
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
  void plotHisto(               DQMStore 				* bei, 
                                MonitorElement                          * theMe,
                                std::string                               theName,
				std::string 				  canvasW,
				std::string 				  canvasH);
  void printModuleHistoList(    DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  void printSummaryHistoList(   DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  void printAlarmList(          DQMStore 				* bei, 
                                std::ostringstream                      & str_val);
  void selectImage(		std::string				& name, 
                                int                                      status);
  void selectImage(		std::string				& name, 
                                std::vector<QReport *>                     & test_map);
  bool goToDir(                 DQMStore                                * bei, 
                                std::string                             & sname);
  bool hasItem(                 std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  std::string getItemValue(     std::multimap<std::string, std::string> & req_map,
	                        std::string                               item_name);
  MonitorElement* getModuleME(  DQMStore                                * bei, 
                                std::string                               me_name);
  void setCanvasMessage(        const std::string                       & error_string);
  void createDummiesFromLayout();  
  void   fillErrorOverviewHistos(DQMStore                               * bei,
				 TH1F                                   * errorHisto,
				 std::string                            & subDet,
				 std::vector<int>                       & hotModuleList);
  int    computeCode(            DQMStore                               * bei,
				 std::string                            & path);
  int    computeSourceCode(      std::string                            & source);
  void   fillPaveTextForErrorCode(TPaveText                             * pave);
  void   coloredHotModules(      TH1F                                   * histo,
				 std::vector<int>                       & binList,
				 int                                      range,
				 int                                      color);
  void setSubDetAxisDrawing(   std::string                                detector, 
                               TH1F                                     * histo);
  void setLines(               MonitorElement                           * me, 
                               std::string                              & meName, 
			       double                                   & ymin, 
			       double                                   & ymax, 
			       double                                   & warning, 
			       double                                   & error, 
			       double                                   & channelFraction);
  
  
  std::ostringstream                     pictureBuffer_ ;
  std::map<std::string, std::string>     namedPictureBuffer ;
  
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
  
};
#endif
