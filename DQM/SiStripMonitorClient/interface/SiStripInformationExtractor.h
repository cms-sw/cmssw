#ifndef _SiStripInformationExtractor_h_
#define _SiStripInformationExtractor_h_

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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

class SiStripInformationExtractor {

 public:

  SiStripInformationExtractor();
 ~SiStripInformationExtractor();

  void readModuleAndHistoList(DaqMonitorBEInterface* bei,xgi::Output * out);
  void plotSingleModuleHistos(DaqMonitorBEInterface * bei,
                      std::multimap<std::string, std::string>& req_map);
  void plotGlobalHistos(DaqMonitorBEInterface * bei,
                      std::multimap<std::string, std::string>& req_map);
  void plotHistosFromPath(DaqMonitorBEInterface * bei,std::multimap<std::string, std::string>& req_map);
  void plotHistosFromLayout(DaqMonitorBEInterface * bei);
  void plotTrackerMapHistos(DaqMonitorBEInterface* bei, std::multimap<std::string, std::string>& req_map);
  const std::ostringstream& getImage() const;
  void readSummaryHistoTree(DaqMonitorBEInterface* bei, std::string& str_name, 
                xgi::Output * out);
  void readAlarmTree(DaqMonitorBEInterface* bei, std::string& str_name, 
                xgi::Output * out);
 
  void readStatusMessage(DaqMonitorBEInterface* bei, std::string& path,xgi::Output * out);
  void readGlobalHistoList(DaqMonitorBEInterface* bei, xgi::Output * out);
  void readLayoutNames(xgi::Output * out);


 private:

  void readConfiguration();
  void fillModuleAndHistoList(DaqMonitorBEInterface * bei,
        std::vector<std::string>& modules, std::vector<std::string>& histos);
  void fillGlobalHistoList(DaqMonitorBEInterface * bei, std::vector<std::string>& histos);
  void selectSingleModuleHistos(DaqMonitorBEInterface * bei,  std::string mid, 
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
  bool goToDir(DaqMonitorBEInterface* bei, std::string& sname);
  void printSummaryHistoList(DaqMonitorBEInterface* bei, std::ostringstream& str_val);
  void printAlarmList(DaqMonitorBEInterface * bei, std::ostringstream& str_val);
  void selectImage(std::string& name, int status);
  void selectImage(std::string& name, dqm::qtests::QR_map& test_map);
  void selectGlobalHistos(DaqMonitorBEInterface * bei, std::vector<std::string>& names, std::vector<MonitorElement*>& mes);
  void defineZone(int nhist, int& ncol, int & now);  
  void setCanvasMessage(const std::string& error_string);
  void createDummiesFromLayout();
  void setDrawingOption(TH1* hist, float xlow=-1.0, float xhigh=-1.0);

  std::ostringstream pictureBuffer_;
  SiStripLayoutParser* layoutParser_;

  std::map<std::string, std::vector< std::string > > layoutMap;
  TCanvas* canvas_;
  bool  readReference_;
};
#endif
