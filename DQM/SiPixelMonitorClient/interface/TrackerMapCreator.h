#ifndef _TrackerMapCreator_h_
#define _TrackerMapCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class TrackerMapCreator {

 public:

  TrackerMapCreator(std::string themEName);
 ~TrackerMapCreator();

  void create(MonitorUserInterface* mui, std::vector<std::string>& me_names, std::string themEName);

 private:
  MonitorElement* getTkMapMe(MonitorUserInterface* mui,std::string& me_name,int ndet);

  void paintTkMap(int det_id, std::map<MonitorElement*, int>& me_map);

  TrackerMap* trackerMap;
  
  std::string mEName ;
  
};
#endif
