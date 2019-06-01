#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TTree.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class DQMStore;
class TkDetMap;
class TrackerTopology;
class MonitorElement;
namespace edm {
  class EventSetup;
}

class SiStripTrackerMapCreator {
public:
  SiStripTrackerMapCreator(edm::EventSetup const& eSetup);
  bool readConfiguration();

  void create(edm::ParameterSet const& tkmapPset,
              DQMStore& dqm_store,
              std::string const& htype,
              edm::EventSetup const& eSetup);
  void createForOffline(edm::ParameterSet const& tkmapPset,
                        DQMStore& dqm_store,
                        std::string& htype,
                        edm::EventSetup const& eSetup);
  void createInfoFile(std::vector<std::string> const& map_names,
                      TTree* tkinfo_tree,
                      DQMStore& dqm_store,
                      std::vector<uint32_t> const& detidList);

private:
  void paintTkMapFromAlarm(uint32_t det_id,
                           const TrackerTopology* tTopo,
                           DQMStore& dqm_store,
                           bool isBad,
                           std::map<unsigned int, std::string>& badmodmap);
  void setTkMapFromHistogram(DQMStore& dqm_store, std::string const& htype, edm::EventSetup const& eSetup);
  void setTkMapFromAlarm(DQMStore& dqm_store, edm::EventSetup const& eSetup);
  uint16_t getDetectorFlagAndComment(DQMStore* dqm_store,
                                     uint32_t det_id,
                                     TrackerTopology const* tTopo,
                                     std::ostringstream& comment);

  void paintTkMapFromHistogram(MonitorElement const* me,
                               std::string const& map_type,
                               std::vector<std::pair<float, uint32_t>>* topNmodVec);
  void setTkMapRange(std::string const& map_type);
  void setTkMapRangeOffline();
  uint16_t getDetectorFlag(uint32_t const det_id) {
    return detFlag_.find(det_id) != detFlag_.end() ? detFlag_[det_id] : 0;
  }
  void printBadModuleList(std::map<unsigned int, std::string> const& badmodmap, edm::EventSetup const& eSetup);
  void printTopModules(std::vector<std::pair<float, uint32_t>>& topNmodVec, edm::EventSetup const& eSetup);

  std::unique_ptr<TrackerMap> trackerMap_{nullptr};
  std::string sRunNumber;
  std::string tkMapName_;
  std::string stripTopLevelDir_{};

  float tkMapMax_;
  float tkMapMin_;
  float meanToMaxFactor_{2.5};
  bool useSSQuality_;
  bool ResidualsRMS_;
  std::string ssqLabel_;
  int nDet_;
  TkDetMap const* tkDetMap_;
  edm::EventSetup const& eSetup_;
  edm::ESHandle<SiStripDetCabling> detCabling_;
  DetId cachedDetId_{};
  int16_t cachedLayer_{};
  std::map<uint32_t, uint16_t> detFlag_;
  TkLayerMap::XYbin cachedXYbin_;
  bool topModules_;
  uint32_t numTopModules_;
  std::string topModLabel_;
};
#endif
