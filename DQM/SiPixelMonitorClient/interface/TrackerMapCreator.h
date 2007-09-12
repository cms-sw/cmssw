#ifndef _TrackerMapCreator_h_
#define _TrackerMapCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMap.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelInformationExtractor;

class TrackerMapCreator {

 public:

  TrackerMapCreator(         std::string            themEName,
                             std::string            theTKType);
 ~TrackerMapCreator();

  //void create(               MonitorUserInterface * mui);
  void create(               DaqMonitorBEInterface * bei);

 private:
  //MonitorElement* getTkMapMe(MonitorUserInterface * mui,
  MonitorElement* getTkMapMe(DaqMonitorBEInterface * bei,
                             std::string          & me_name,
			     int                    ndet);

  void paintTkMap(           MonitorElement       * mE);

  SiPixelInformationExtractor			  * infoExtractor_;
  SiPixelTrackerMap          			  * trackerMap;
  						  
  std::string					    mEName ;
  std::string					    TKType ;
  
  //bool exploreMuiStructure(  MonitorUserInterface * mui) ;
  bool exploreBeiStructure(  DaqMonitorBEInterface * bei) ;
};
#endif
