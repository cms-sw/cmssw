#ifndef _TrackerMapCreator_h_
#define _TrackerMapCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMap.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelInformationExtractor;

class TrackerMapCreator {

 public:

  TrackerMapCreator(std::string themEName);
 ~TrackerMapCreator();

  void create(               MonitorUserInterface     	    * mui);

 private:
  MonitorElement* getTkMapMe(MonitorUserInterface     	    * mui,
                             std::string              	    & me_name,
			     int                      	      ndet);

  void paintTkMap(           MonitorElement                 * mui);

  SiPixelInformationExtractor * infoExtractor_;
  SiPixelTrackerMap           * trackerMap;
  
  std::string mEName ;
  
  bool exploreMuiStructure(MonitorUserInterface* mui) ;
};
#endif
