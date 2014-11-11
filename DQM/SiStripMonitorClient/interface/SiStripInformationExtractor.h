#ifndef _SiStripInformationExtractor_h_
#define _SiStripInformationExtractor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <map>

class SiStripLayoutParser;
class SiStripDetCabling;
class QReport;
class SiStripHistoPlotter;

class SiStripInformationExtractor {
  
 public:
  
  SiStripInformationExtractor();
  ~SiStripInformationExtractor();
  
  void plotHistosFromLayout(DQMStore::IGetter &);
  void createImages(DQMStore::IGetter &);
  
  // removing xdaq deps
 // void getSingleModuleHistos(DQMStore * dqm_store, 
  //     const std::multimap<std::string, std::string>& req_map, xgi::Output * out);
 // void getGlobalHistos(DQMStore* dqm_store, 
  //     const std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  //void getHistosFromPath(DQMStore * dqm_store, 
    //   const std::multimap<std::string, std::string>& req_map, xgi::Output * out);
//  void getTrackerMapHistos(DQMStore* dqm_store, 
  //     const std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  //void getCondDBHistos(DQMStore* dqm_store, bool& plot_flag,
    //   const std::multimap<std::string, std::string>& req_map, xgi::Output * out);

  //void readModuleAndHistoList(DQMStore* dqm_store,std::string& sname, const edm::ESHandle<SiStripDetCabling>& detcabling,xgi::Output * out);
  //void readSummaryHistoTree(DQMStore* dqm_store, std::string& str_name, 
      //          xgi::Output * out);
  //void readAlarmTree(DQMStore* dqm_store, std::string& str_name, 
 //               xgi::Output * out);

  //void readStatusMessage(DQMStore* dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out);
  //void readGlobalHistoList(DQMStore* dqm_store, std::string& dname, xgi::Output * out);
  //void readLayoutNames(DQMStore* dqm_store, xgi::Output * out);

  //void readQTestSummary(DQMStore* dqm_store, std::string type, xgi::Output * out);

  //void readNonGeomHistoTree(DQMStore* dqm_store, std::string& fld_name, xgi::Output * out);

  //void getImage(const std::multimap<std::string, std::string>& req_map, xgi::Output * out);


 private:

  void readConfiguration();

  void getItemList(const std::multimap<std::string, std::string>& req_map,
                   std::string item_name, std::vector<std::string>& items);
  bool hasItem(const std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  std::string getItemValue(const std::multimap<std::string, std::string>& req_map,
	      std::string item_name);
  void printSummaryHistoList(DQMStore::IBooker & , DQMStore::IGetter &, std::ostringstream& str_val);
  void printAlarmList(DQMStore::IBooker & , DQMStore::IGetter &, std::ostringstream& str_val);
  void printNonGeomHistoList(DQMStore::IBooker & , DQMStore::IGetter &, std::ostringstream& str_val);

  void selectImage(std::string& name, int status);
  void selectImage(std::string& name, std::vector<QReport*> & reports);
  void selectImage(DQMStore::IGetter & , std::string& , std::string);
  void selectColor(std::string& col, int status);
  void selectColor(std::string& col, std::vector<QReport*>& reports);

  
  //void setHTMLHeader(xgi::Output * out);
  //void setXMLHeader(xgi::Output * out);
  //void setPlainHeader(xgi::Output * out);


  SiStripLayoutParser* layoutParser_;

  std::map<std::string, std::vector< std::string > > layoutMap;
  std::vector<std::string> subdetVec;
  bool  readReference_;
 

  SiStripHistoPlotter* histoPlotter_;
};
#endif
