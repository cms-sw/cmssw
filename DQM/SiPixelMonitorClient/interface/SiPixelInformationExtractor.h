#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "xgi/include/xgi/Utils.h"
#include "xgi/include/xgi/Method.h"

#include "TCanvas.h"

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

  void readModuleAndHistoList(MonitorUserInterface* mui,xgi::Output * out, bool coll_flag);
  void plotSingleModuleHistos(MonitorUserInterface * mui,
                      std::multimap<std::string, std::string>& req_map);
  const std::ostringstream& getImage() const;
  void readSummaryHistoTree(MonitorUserInterface* mui, 
    std::string& str_name, xgi::Output * out, bool coll_flag);
  void readAlarmTree(MonitorUserInterface* mui, std::string& str_name, 
                xgi::Output * out, bool coll_flag);
  void plotSingleHistogram(MonitorUserInterface * mui,std::multimap<std::string, std::string>& req_map);
  void readStatusMessage(MonitorUserInterface* mui, std::string& path,xgi::Output * out);

 private:

  void fillModuleAndHistoList(MonitorUserInterface * mui,
        std::vector<std::string>& modules, std::vector<std::string>& histos);
  void selectSingleModuleHistos(MonitorUserInterface * mui,  std::string mid, 
          std::vector<std::string>& names, std::vector<MonitorElement*>& mes);
  void getItemList(std::multimap<std::string, std::string>& req_map,
                   std::string item_name, std::vector<std::string>& items);
  bool hasItem(std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  std::string getItemValue(std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  void fillImageBuffer(TCanvas& c1);
  void plotHistos(std::multimap<std::string, std::string>& req_map, 
                  std::vector<MonitorElement*> me_list);
  bool goToDir(MonitorUserInterface* mui, std::string& sname, bool flg);
  void printSummaryHistoList(MonitorUserInterface* mui, std::ostringstream& str_val);
  void printAlarmList(MonitorUserInterface * mui, std::ostringstream& str_val);
  void selectImage(std::string& name, int status);
  void selectImage(std::string& name, dqm::qtests::QR_map& test_map);

  std::ostringstream pictureBuffer_;
};
#endif
