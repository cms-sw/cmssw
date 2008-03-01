#ifndef _SiStripInformationExtractor_h_
#define _SiStripInformationExtractor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "TCanvas.h"
#include "TH1.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <map>

class SiStripLayoutParser;
class SiStripDetCabling;
class DQMStore;
class QReport;
class SiStripInformationExtractor {

 public:

  SiStripInformationExtractor();
 ~SiStripInformationExtractor();

  void readModuleAndHistoList(DQMStore* dqm_store,const edm::ESHandle<SiStripDetCabling>& detcabling,xgi::Output * out);
  void plotSingleModuleHistos(DQMStore * dqm_store,
                      std::multimap<std::string, std::string>& req_map, xgi::Output * out );
  void plotGlobalHistos(DQMStore * dqm_store,
                      std::multimap<std::string, std::string>& req_map, xgi::Output * out );
  void plotHistosFromPath(DQMStore * dqm_store,std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  void plotHistosFromLayout(DQMStore * dqm_store);
  void plotTrackerMapHistos(DQMStore* dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  const std::ostringstream& getImage() const;
  const std::ostringstream& getNamedImage(std::string name);
  void readSummaryHistoTree(DQMStore* dqm_store, std::string& str_name, 
                xgi::Output * out);
  void readAlarmTree(DQMStore* dqm_store, std::string& str_name, 
                xgi::Output * out);
 
  void readStatusMessage(DQMStore* dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  void readGlobalHistoList(DQMStore* dqm_store, std::string& dname, xgi::Output * out);
  void readLayoutNames(std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  const std::ostringstream& getIMGCImage(DQMStore * dqm_store, std::multimap<std::string, std::string>& req_map);

  void readQTestSummary(DQMStore* dqm_store, std::string type, const edm::ESHandle<SiStripDetCabling>& detcabling, xgi::Output * out);


 private:

  void readConfiguration();
  void fillModuleAndHistoList(DQMStore * dqm_store,
        std::vector<std::string>& modules, std::vector<std::string>& histos);
  void fillGlobalHistoList(DQMStore * dqm_store, std::vector<std::string>& histos);
  void selectSingleModuleHistos(DQMStore * dqm_store,  std::string mid, 
          std::vector<std::string>& names, std::vector<MonitorElement*>& mes);
  void getItemList(std::multimap<std::string, std::string>& req_map,
                   std::string item_name, std::vector<std::string>& items);
  bool hasItem(std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  std::string getItemValue(std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  void fillImageBuffer();
  void plotHistos(std::multimap<std::string, std::string>& req_map, 
                  std::vector<MonitorElement*> me_list, bool sflag);
  void plotHisto(std::multimap<std::string, std::string>& req_map, 
                  MonitorElement* me, bool sflag);
  bool goToDir(DQMStore* dqm_store, std::string& sname);
  void printSummaryHistoList(DQMStore* dqm_store, std::ostringstream& str_val);
  void printAlarmList(DQMStore * dqm_store, std::ostringstream& str_val);
  void selectImage(std::string& name, int status);
  void selectImage(std::string& name, std::vector<QReport*> & reports);
  void selectGlobalHistos(DQMStore * dqm_store, std::string dname, 
             std::vector<std::string>& names, std::vector<MonitorElement*>& mes);
  void defineZone(int nhist, int& ncol, int & now);  
  void setCanvasMessage(const std::string& error_string);
  void createDummiesFromLayout();
  void setDrawingOption(TH1* hist, float xlow=-1.0, float xhigh=-1.0);
  void fillNamedImageBuffer(std::string name);
  bool hasNamedImage(std::string name);
  void setCanvasDimension(std::multimap<std::string, std::string>& req_map);
  

  std::ostringstream pictureBuffer_;
  std::map<std::string, std::string>       namedPictureBuffer_;

  SiStripLayoutParser* layoutParser_;

  std::map<std::string, std::vector< std::string > > layoutMap;
  TCanvas* canvas_;
  bool  readReference_;
 
  int canvasWidth;
  int canvasHeight;
};
#endif
