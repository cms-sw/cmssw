#ifndef _SiPixelTrackerMapCreator_h_
#define _SiPixelTrackerMapCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMap.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelInformationExtractor;

class SiPixelTrackerMapCreator {

 public:

  SiPixelTrackerMapCreator(  std::string            themEName,
                             std::string            theTKType,
			     bool                   offlineXMLfile);
 ~SiPixelTrackerMapCreator();

  void create(               DQMStore * bei);

 private:

  MonitorElement* getTkMapMe(DQMStore * bei,
                             std::string          & me_name,
			     int                    ndet);

  void paintTkMap(           MonitorElement       * mE);

  SiPixelInformationExtractor			  * infoExtractor_;
  SiPixelTrackerMap          			  * trackerMap;
  						  
  std::string					    mEName ;
  std::string					    TKType ;
  bool                                              offlineXMLfile_;
  
  bool exploreBeiStructure(  DQMStore * bei) ;
};
#endif
