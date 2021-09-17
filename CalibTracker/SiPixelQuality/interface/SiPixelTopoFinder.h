#ifndef SiPixelTopoFinder_H
#define SiPixelTopoFinder_H
// -*- C++ -*-
//
// Class:      SiPixelTopoFinder
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

class SiPixelTopoFinder {
public:
  SiPixelTopoFinder();
  ~SiPixelTopoFinder();

  void init(const TrackerGeometry* trackerGeometry,
            const TrackerTopology* trackerTopology,
            const SiPixelFedCablingMap* siPixelFedCablingMap);

  std::vector<int> getDetIds() const { return fDetIds_; }

  std::map<int, std::pair<int, int>> getSensors() const { return fSensors_; }

  std::map<int, std::pair<int, int>> getSensorLayout() const { return fSensorLayout_; }

  std::unordered_map<uint32_t, unsigned int> getFedIds() const { return fFedIds_; }

  std::map<int, std::map<int, int>> getRocIds() const { return fRocIds_; }

private:
  // initialize with nullptr
  int phase_ = -1;

  const TrackerTopology* tkTopo_ = nullptr;
  const TrackerGeometry* tkGeom_ = nullptr;
  const SiPixelFedCablingMap* cablingMap_ = nullptr;

  // List of <int> DetIds
  std::vector<int> fDetIds_;
  // ROC size (number of row, number of columns for each det id)
  std::map<int, std::pair<int, int>> fSensors_;
  // the roc layout on a module
  std::map<int, std::pair<int, int>> fSensorLayout_;
  // fedId as a function of detId
  std::unordered_map<uint32_t, unsigned int> fFedIds_;
  // map the index ROC to rocId
  std::map<int, std::map<int, int>> fRocIds_;

  // conversion between online(local, per-ROC) row/column and offline(global, per-Module) row/column
  void onlineRocColRow(const DetId& detId,
                       const SiPixelFedCablingMap* cablingMap,
                       int fedId,
                       int offlineRow,
                       int offlineCol,
                       int& roc,
                       int& row,
                       int& col);

  int indexROC(int irow, int icol, int nROCcolumns);

  // some helper function for pixel naming
  int quadrant(const DetId& detid);
  int side(const DetId& detid);
  int half(const DetId& detid);
};

#endif
