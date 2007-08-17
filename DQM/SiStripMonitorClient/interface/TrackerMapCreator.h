#ifndef _TrackerMapCreator_h_
#define _TrackerMapCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CommonTools/TrackerMap/interface/FedTrackerMap.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class DaqMonitorBEInterface;
class SiStripDetCabling;
class SiStripFedCabling;

class TrackerMapCreator {

 public:

  TrackerMapCreator();
 ~TrackerMapCreator();
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

  TrackerMap* trackerMap_;
  FedTrackerMap* fedTrackerMap_;
  std::vector<std::string> meNames_;
  std::string tkMapName_;
  int tkMapFrequency_;
};
#endif
