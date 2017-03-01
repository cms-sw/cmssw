#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
//#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <TTree.h>

class DQMStore;
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
              DQMStore* dqm_store, std::string& htype, const edm::EventSetup& eSetup);
  void createForOffline(const edm::ParameterSet & tkmapPset, 
			DQMStore* dqm_store, std::string& htype, const edm::EventSetup& eSetup);
  void createInfoFile(std::vector<std::string> map_names, TTree* tkinfo_tree, DQMStore* dqm_store, std::vector<uint32_t> detidList);


 private:

  void paintTkMapFromAlarm(uint32_t det_id, const TrackerTopology* tTopo,
                           DQMStore* dqm_store, bool isBad=false, std::map<unsigned int,std::string>* badmodmap=0);
  void paintTkMapFromHistogram(DQMStore* dqm_store, MonitorElement* me, std::string& map_type, std::vector<std::pair<float,uint32_t> >* topNmodVec);
  void setTkMapFromHistogram(DQMStore* dqm_store, std::string& htype, const edm::EventSetup& eSetup);
  void setTkMapFromAlarm(DQMStore* dqm_store,  const edm::EventSetup& eSetup);
  void setTkMapRange(std::string& map_type);
  void setTkMapRangeOffline();
  uint16_t getDetectorFlagAndComment(DQMStore* dqm_store, uint32_t det_id, const TrackerTopology* tTopo, std::ostringstream& comment);
  uint16_t getDetectorFlag(uint32_t det_id) { return detflag_.find(det_id) != detflag_.end() ? detflag_[det_id] : 0; }
  void printBadModuleList(std::map<unsigned int,std::string>* badmodmap, const edm::EventSetup& eSetup);
  void printTopModules(std::vector<std::pair<float,uint32_t> >* topNmodVec, const edm::EventSetup& eSetup);


  TrackerMap* trackerMap_;
  std::string sRunNumber; //LG
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
  //  SiStripPsuDetIdMap psumap_;
  uint32_t cached_detid;
  int16_t cached_layer;
  std::map<uint32_t, uint16_t> detflag_;
  TkLayerMap::XYbin cached_XYbin;
  bool topModules;
  uint32_t numTopModules;
  std::string topModLabel;
};
#endif
