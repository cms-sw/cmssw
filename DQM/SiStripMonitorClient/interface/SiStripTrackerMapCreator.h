#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include <map>
#include <vector>
#include <string>

class DQMStore;
class TkDetMap;
class TrackerMap;
class MonitorElement;

class SiStripTrackerMapCreator {

 public:

  SiStripTrackerMapCreator();
 ~SiStripTrackerMapCreator();
  bool readConfiguration();

  void create(const edm::ParameterSet & tkmapPset, 
	      const edm::ESHandle<SiStripFedCabling>& fedcabling, 
              DQMStore* dqm_store, std::string& htype);
  void createForOffline(const edm::ParameterSet & tkmapPset, 
			const edm::ESHandle<SiStripFedCabling>& fedcabling,
			DQMStore* dqm_store, std::string& htype);


 private:

  void paintTkMapFromAlarm(uint32_t det_id, DQMStore* dqm_store);
  void paintTkMapFromHistogram(DQMStore* dqm_store, MonitorElement* me, std::string& map_type);
  void setTkMapFromHistogram(DQMStore* dqm_store, std::string& htype);
  void setTkMapRange(std::string& map_type);
  void setTkMapRangeOffline();
  uint16_t getDetectorFlagAndComment(DQMStore* dqm_store, uint32_t det_id, std::ostringstream& comment);

  TrackerMap* trackerMap_;
  std::string tkMapName_;

  float tkMapMax;
  float tkMapMin;
  bool tkMapLog;
  int   nDet;
  TkDetMap* tkDetMap_;
};
#endif
