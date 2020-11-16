#ifndef _SiPixelInformationExtractor_h_
#define _SiPixelInformationExtractor_h_

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelLayoutParser.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "TCanvas.h"
#include "TF1.h"
#include "TGaxis.h"
#include "TH2F.h"
#include "TPaveText.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

class SiPixelEDAClient;
class SiPixelWebInterface;
class SiPixelInformationExtractor {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  SiPixelInformationExtractor(bool offlineXMLfile);
  ~SiPixelInformationExtractor();

  void computeStatus(MonitorElement *mE, double &colorValue, std::pair<double, double> &norm);
  void getNormalization(MonitorElement *mE, std::pair<double, double> &norm, std::string theMEType);
  void getNormalization2D(MonitorElement *mE,
                          std::pair<double, double> &normX,
                          std::pair<double, double> &normY,
                          std::string theMEType);
  int getDetId(MonitorElement *mE);
  std::string getMEType(MonitorElement *mE);

  void readConfiguration();
  bool readConfiguration(std::map<std::string, std::vector<std::string>> &layoutMap,
                         std::map<std::string, std::map<std::string, std::string>> &qtestsMap,
                         std::map<std::string, std::vector<std::string>> &meQTestsMap);

  void bookNoisyPixels(DQMStore::IBooker &iBooker, float noiseRate, bool Tier0Flag);

  void findNoisyPixels(DQMStore::IBooker &iBooker,
                       DQMStore::IGetter &iGetter,
                       bool init,
                       float noiseRate,
                       int noiseRateDenominator,
                       const SiPixelFedCablingMap *theCablingMap);

private:
  void getItemList(const std::multimap<std::string, std::string> &req_map,
                   std::string item_name,
                   std::vector<std::string> &items);
  bool hasItem(std::multimap<std::string, std::string> &req_map, std::string item_name);
  std::string getItemValue(const std::multimap<std::string, std::string> &req_map, std::string item_name);
  std::string getItemValue(std::multimap<std::string, std::string> &req_map, std::string item_name);
  void createDummiesFromLayout();
  void selectImage(std::string &name, int status);
  void selectImage(std::string &name, std::vector<QReport *> &reports);
  void selectColor(std::string &col, int status);
  void selectColor(std::string &col, std::vector<QReport *> &reports);

  int alarmCounter_;

  SiPixelConfigParser *configParser_;
  SiPixelConfigWriter *configWriter_;
  SiPixelActionExecutor *actionExecutor_;
  SiPixelLayoutParser *layoutParser_;

  std::map<std::string, std::vector<std::string>> layoutMap;
  std::map<std::string, std::map<std::string, std::string>> qtestsMap;
  std::map<std::string, std::vector<std::string>> meQTestsMap;

  bool readReference_;
  bool readQTestMap_;
  bool readMeMap_;
  bool flagHotModule_;
  bool offlineXMLfile_;

  int count;
  int errcount;
  bool gotDigis;

  std::ofstream myfile_;
  int nevents_;
  std::map<uint32_t, std::vector<std::pair<std::pair<int, int>, float>>> noisyDetIds_;
  bool endOfModules_;

  MonitorElement *EventRateBarrelPixels;
  MonitorElement *EventRateEndcapPixels;

  MonitorElement *EndcapNdigisFREQProjection;
  MonitorElement *BarrelNdigisFREQProjection;
};
#endif
