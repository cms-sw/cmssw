#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
//#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQMServices/Core/interface/DQMStore.h"


#include <fstream>
#include <map>
#include <vector>
#include <string>

class TkDetMap;
class TrackerMap;
class TrackerTopology;
class MonitorElement;
namespace edm { class EventSetup; }

class SiStripTrackerMapCreator {

 public:

  //  SiStripTrackerMapCreator();
  SiStripTrackerMapCreator(const edm::EventSetup& eSetup);
 ~SiStripTrackerMapCreator();
  bool readConfiguration();

  void create(const edm::ParameterSet & tkmapPset, 
              DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string& htype, edm::ESHandle<SiStripQuality> & ssq);
  void createForOffline(const edm::ParameterSet & tkmapPset, 
			DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string& htype, edm::ESHandle<SiStripQuality> & ssq);


 private:

  void paintTkMapFromAlarm(uint32_t det_id, const TrackerTopology* tTopo,
                           DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, bool isBad=false, std::map<unsigned int,std::string>* badmodmap=0);
  void paintTkMapFromHistogram(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, MonitorElement* me, std::string& map_type);
  void setTkMapFromHistogram(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string& htype);
  void setTkMapFromAlarm(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,  edm::ESHandle<SiStripQuality> ssq);
  void setTkMapRange(std::string& map_type);
  void setTkMapRangeOffline();
  uint16_t getDetectorFlagAndComment(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, uint32_t det_id, const TrackerTopology* tTopo, std::ostringstream& comment);
  void printBadModuleList(std::map<unsigned int,std::string>* badmodmap);

  TrackerMap* trackerMap_;
  std::string tkMapName_;
  std::string stripTopLevelDir_;

  float tkMapMax_;
  float tkMapMin_;
  float meanToMaxFactor_;
  bool useSSQuality_;
  std::string ssqLabel_;
  int   nDet;
  TkDetMap* tkDetMap_;
  const edm::EventSetup& eSetup_;
  edm::ESHandle< SiStripDetCabling > detcabling_;
  const TrackerTopology* tTopo;
  //  SiStripPsuDetIdMap psumap_;
  uint32_t cached_detid;
  int16_t cached_layer;
  TkLayerMap::XYbin cached_XYbin;
};
#endif
