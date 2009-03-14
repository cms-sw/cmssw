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

class TrackerMap;
class SiStripTrackerMapCreator {

 public:

  SiStripTrackerMapCreator();
 ~SiStripTrackerMapCreator();
  bool readConfiguration();

  void create(const edm::ParameterSet & tkmapPset, 
	      const edm::ESHandle<SiStripFedCabling>& fedcabling, 
              DQMStore* dqm_store, std::string& htype);

 private:

  void paintTkMapFromAlarm(int det_id, DQMStore* dqm_store);
  void paintTkMapFromHistogram(int det_id, DQMStore* dqm_store, std::string& map_type);

  TrackerMap* trackerMap_;
  std::string tkMapName_;
};
#endif
