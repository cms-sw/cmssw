#ifndef DQM_SiStripCommon_TKHistoMap_h
#define DQM_SiStripCommon_TKHistoMap_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include <string>

class TkHistoMap{

 public:
  TkHistoMap(std::string path, std::string MapName);
  ~TkHistoMap();
  
  void fill(uint32_t& detid,float& value);
  void setBinContent(uint32_t& detid,float& value);

 private:

  void createTkHistoMap(std::string& path, std::string& MapName);

  DQMStore* dqmStore_;
  TkDetMap* tkdetmap_;
  typedef std::vector<MonitorElement*> tkHistoMapType;
  tkHistoMapType tkHistoMap;
};

#endif
