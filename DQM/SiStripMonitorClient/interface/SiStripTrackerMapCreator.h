#ifndef _SiStripTrackerMapCreator_h_
#define _SiStripTrackerMapCreator_h_

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TTree.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TkDetMap;
class TrackerTopology;
class SiStripQuality;
class GeometricDet;

class SiStripTrackerMapCreator {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripTrackerMapCreator(const SiStripDetCabling* detCabling, const TkDetMap* tkDetMap, const TrackerTopology* tTopo);
  bool readConfiguration();

  void create(edm::ParameterSet const& tkmapPset, DQMStore& dqm_store, std::string const& htype);
  void createForOffline(edm::ParameterSet const& tkmapPset,
                        DQMStore& dqm_store,
                        std::string& htype,
                        const SiStripQuality*);
  void createInfoFile(std::vector<std::string> const& map_names,
                      TTree* tkinfo_tree,
                      DQMStore& dqm_store,
                      const GeometricDet* geomDet);

private:
  void paintTkMapFromAlarm(uint32_t det_id,
                           DQMStore& dqm_store,
                           bool isBad,
                           std::map<unsigned int, std::string>& badmodmap);
  void setTkMapFromHistogram(DQMStore& dqm_store, std::string const& htype);
  void setTkMapFromAlarm(DQMStore& dqm_store, const SiStripQuality* stripQuality);
  uint16_t getDetectorFlagAndComment(DQMStore* dqm_store, uint32_t det_id, std::ostringstream& comment);

  void paintTkMapFromHistogram(MonitorElement const* me,
                               std::string const& map_type,
                               std::vector<std::pair<float, uint32_t>>* topNmodVec);
  void setTkMapRange(std::string const& map_type);
  void setTkMapRangeOffline();
  uint16_t getDetectorFlag(uint32_t const det_id) {
    return detFlag_.find(det_id) != detFlag_.end() ? detFlag_[det_id] : 0;
  }
  void printBadModuleList(std::map<unsigned int, std::string> const& badmodmap);
  void printTopModules(std::vector<std::pair<float, uint32_t>>& topNmodVec);

  std::unique_ptr<TrackerMap> trackerMap_{nullptr};
  std::string sRunNumber;
  std::string tkMapName_;
  std::string stripTopLevelDir_{};

  float tkMapMax_;
  float tkMapMin_;
  float meanToMaxFactor_{2.5};
  bool ResidualsRMS_;
  int nDet_;
  const SiStripDetCabling* detCabling_;
  TkDetMap const* tkDetMap_;
  const TrackerTopology* tTopo_;
  DetId cachedDetId_{};
  int16_t cachedLayer_{};
  std::map<uint32_t, uint16_t> detFlag_;
  TkLayerMap::XYbin cachedXYbin_;
  bool topModules_;
  uint32_t numTopModules_;
  std::string topModLabel_;
};
#endif
