#ifndef DQM_SiStripCommon_TKHistoMap_h
#define DQM_SiStripCommon_TKHistoMap_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include <string>

class TkHistoMap{

  typedef std::vector<MonitorElement*> tkHistoMapType;

 public:
  TkHistoMap(std::string path, std::string MapName, float baseline=0, bool mechanicalView=false);
  ~TkHistoMap();

  MonitorElement* getMap(short layerNumber){return tkHistoMap_[layerNumber];};
  tkHistoMapType& getAllMaps(){return tkHistoMap_;};

  void fill(uint32_t& detid,float value);
  void setBinContent(uint32_t& detid,float value);
  void add(uint32_t& detid,float value);

  void save(std::string filename);
  void saveAsCanvas(std::string filename,std::string options="", std::string mode="RECREATE");

 private:

  void createTkHistoMap(std::string& path, std::string& MapName, float& baseline, bool mechanicalView);

  DQMStore* dqmStore_;
  TkDetMap* tkdetmap_;
  tkHistoMapType tkHistoMap_;
  int HistoNumber;
  std::string MapName_;
};

#endif
