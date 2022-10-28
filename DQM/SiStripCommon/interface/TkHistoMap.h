#ifndef DQM_SiStripCommon_TKHistoMap_h
#define DQM_SiStripCommon_TKHistoMap_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include <string>

class TkHistoMap {
protected:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef std::vector<MonitorElement*> tkHistoMapVect;

public:
  TkHistoMap(const TkDetMap* tkDetMap,
             DQMStore::IBooker& ibooker,
             const std::string& path,
             const std::string& MapName,
             float baseline = 0,
             bool mechanicalView = false,
             bool isTH2F = false);
  TkHistoMap(const TkDetMap* tkDetMap,
             const std::string& path,
             const std::string& MapName,
             float baseline = 0,
             bool mechanicalView = false);
  TkHistoMap(const TkDetMap* tkDetMap,
             const std::string& path,
             const std::string& MapName,
             float baseline,
             bool mechanicalView,
             bool isTH2F);
  TkHistoMap(const TkDetMap* tkDetMap);
  ~TkHistoMap() = default;

  void loadServices();

  void loadTkHistoMap(const std::string& path, const std::string& MapName, bool mechanicalView = false);

  MonitorElement* getMap(short layerNumber) { return tkHistoMap_[layerNumber]; };
  const std::vector<MonitorElement*>& getAllMaps() const { return tkHistoMap_; };
  std::vector<MonitorElement*>& getAllMaps() { return tkHistoMap_; };

  float getValue(DetId detid);
  float getEntries(DetId detid);
  DetId getDetId(const std::string& title, int ix, int iy) {
    return getDetId(getLayerNum(getLayerName(title)), ix, iy);
  }
  DetId getDetId(int layer, int ix, int iy) { return tkdetmap_->getDetFromBin(layer, ix, iy); }
  DetId getDetId(const MonitorElement* ME, int ix, int iy) { return getDetId(ME->getTitle(), ix, iy); }
  std::string getLayerName(std::string title) { return title.erase(0, MapName_.size() + 1); }
  uint16_t getLayerNum(const std::string& layerName) { return tkdetmap_->getLayerNum(layerName); }

  void fillFromAscii(const std::string& filename);
  void fill(DetId detid, float value);
  void setBinContent(DetId detid, float value);
  void add(DetId detid, float value);

  void dumpInTkMap(TrackerMap* tkmap,
                   bool dumpEntries = false);  //dumpEntries==true? (dump entries) : (dump mean values)
  void save(const std::string& filename);
  void saveAsCanvas(const std::string& filename, const std::string& options = "", const std::string& mode = "RECREATE");

private:
  void load(const TkDetMap* tkDetMap,
            const std::string& path,
            float baseline,
            bool mechanicalView,
            bool isTH2F,
            bool createTkMap = true);

  void createTkHistoMap(DQMStore::IBooker& ibooker,
                        const std::string& path,
                        const std::string& MapName,
                        float baseline,
                        bool mechanicalView);

  std::string folderDefinition(DQMStore::IBooker& ibooker,
                               std::string folder,
                               const std::string& MapName,
                               int layer,
                               bool mechanicalView,
                               std::string& fullName);

  DQMStore* dqmStore_{nullptr};

  const TkDetMap* tkdetmap_;
  DetId cached_detid;
  int16_t cached_layer;
  TkLayerMap::XYbin cached_XYbin;
  std::vector<MonitorElement*> tkHistoMap_;
  int HistoNumber;
  std::string MapName_;
  bool isTH2F_;
};

#endif
