#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <fstream>
#include <map>
#include <vector>
#include <string>

class DaqMonitorBEInterface;

class FedTrackerMap;
class SiStripTrackerMap;
class SiStripTrackerMapCreator {

 public:

  SiStripTrackerMapCreator();
 ~SiStripTrackerMapCreator();
  bool readConfiguration();

  void create(DaqMonitorBEInterface* bei);
  void create(const edm::ESHandle<SiStripFedCabling> fedcabling, DaqMonitorBEInterface* bei);
  void createFedTkMap(const edm::ESHandle<SiStripFedCabling> fedcabling, DaqMonitorBEInterface* bei);
  int getFrequency() { return tkMapFrequency_;}
  int getMENames(std::vector< std::string>& me_names);


 private:
  MonitorElement* getTkMapMe(DaqMonitorBEInterface* bei, std::string& me_name, int ndet);

  void paintTkMap(int det_id, std::map<MonitorElement*, int>& me_map);
  void paintFedTkMap(int fed_id, int fed_ch, std::map<MonitorElement*, int>& me_map);

  SiStripTrackerMap* trackerMap_;
  FedTrackerMap* fedTrackerMap_;
  std::vector<std::string> meNames_;
  std::string tkMapName_;
  int tkMapFrequency_;
};
#endif
