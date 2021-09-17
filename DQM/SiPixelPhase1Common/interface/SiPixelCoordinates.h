#ifndef SiPixelCoordinates_h
#define SiPixelCoordinates_h
// -*- C++ -*-
//
// Class:      SiPixelCoordinates
//
// This class provides floating point numbers for
// digis, clusters and hits that can be used to
// easily plot various geometry related histograms
//
// Online and Offline conventions are kept for the variables
// An additional ]0, +1[ or ]-0.5, +0.5[ is added depending on
// the location of digi/cluster/hit on the module, so
// the variables provided are roughly monotonic vs. global
// coordinates (except that overlaps are removed), eg:
// global     z: module, disk, disk_ring (large division)
// global r-phi: ladder, blade, blade_panel
// global     r:         ring, disk_ring (small division)
//
// Original Author: Janos Karancsi

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include <cstdint>
#include <unordered_map>
#include <utility>

class SiPixelCoordinates {
public:
  SiPixelCoordinates();
  SiPixelCoordinates(int);
  virtual ~SiPixelCoordinates();

  void init(const TrackerTopology*, const TrackerGeometry*, const SiPixelFedCablingMap*);

  // Integers
  int quadrant(const DetId&);
  int side(const DetId&);
  int module(const DetId&);
  // barrel specific
  int layer(const DetId&);
  int sector(const DetId&);
  int ladder(const DetId&);
  int signed_ladder(const DetId&);
  int signed_module(const DetId&);
  int half(const DetId&);
  int outer(const DetId&);
  int flipped(const DetId&);
  // endcap specific
  int disk(const DetId&);
  int signed_disk(const DetId&);
  int panel(const DetId&);
  int ring(const DetId&);
  int blade(const DetId&);
  int signed_blade(const DetId&);

  unsigned int fedid(const DetId&);

  int channel(const DetId&, const std::pair<int, int>&);
  int channel(const DetId&, const PixelDigi*);
  int channel(const DetId&, const SiPixelCluster*);
  int channel(const SiPixelRecHit*);
  int channel(const TrackingRecHit*);

  int roc(const DetId&, const std::pair<int, int>&);
  int roc(const DetId&, const PixelDigi*);
  int roc(const DetId&, const SiPixelCluster*);
  int roc(const SiPixelRecHit*);
  int roc(const TrackingRecHit*);

  // Floats (coordinates)
  float module_coord(const DetId&, const std::pair<int, int>&);
  float module_coord(const DetId&, const PixelDigi*);
  float module_coord(const DetId&, const SiPixelCluster*);
  float module_coord(const SiPixelRecHit*);
  float module_coord(const TrackingRecHit*);

  float signed_module_coord(const DetId&, const std::pair<int, int>&);
  float signed_module_coord(const DetId&, const PixelDigi*);
  float signed_module_coord(const DetId&, const SiPixelCluster*);
  float signed_module_coord(const SiPixelRecHit*);
  float signed_module_coord(const TrackingRecHit*);

  float ladder_coord(const DetId&, const std::pair<int, int>&);
  float ladder_coord(const DetId&, const PixelDigi*);
  float ladder_coord(const DetId&, const SiPixelCluster*);
  float ladder_coord(const SiPixelRecHit*);
  float ladder_coord(const TrackingRecHit*);

  float signed_ladder_coord(const DetId&, const std::pair<int, int>&);
  float signed_ladder_coord(const DetId&, const PixelDigi*);
  float signed_ladder_coord(const DetId&, const SiPixelCluster*);
  float signed_ladder_coord(const SiPixelRecHit*);
  float signed_ladder_coord(const TrackingRecHit*);

  float ring_coord(const DetId&, const std::pair<int, int>&);
  float ring_coord(const DetId&, const PixelDigi*);
  float ring_coord(const DetId&, const SiPixelCluster*);
  float ring_coord(const SiPixelRecHit*);
  float ring_coord(const TrackingRecHit*);

  float disk_coord(const DetId&, const std::pair<int, int>&);
  float disk_coord(const DetId&, const PixelDigi*);
  float disk_coord(const DetId&, const SiPixelCluster*);
  float disk_coord(const SiPixelRecHit*);
  float disk_coord(const TrackingRecHit*);

  float signed_disk_coord(const DetId&, const std::pair<int, int>&);
  float signed_disk_coord(const DetId&, const PixelDigi*);
  float signed_disk_coord(const DetId&, const SiPixelCluster*);
  float signed_disk_coord(const SiPixelRecHit*);
  float signed_disk_coord(const TrackingRecHit*);

  float disk_ring_coord(const DetId&, const std::pair<int, int>&);
  float disk_ring_coord(const DetId&, const PixelDigi*);
  float disk_ring_coord(const DetId&, const SiPixelCluster*);
  float disk_ring_coord(const SiPixelRecHit*);
  float disk_ring_coord(const TrackingRecHit*);

  float signed_disk_ring_coord(const DetId&, const std::pair<int, int>&);
  float signed_disk_ring_coord(const DetId&, const PixelDigi*);
  float signed_disk_ring_coord(const DetId&, const SiPixelCluster*);
  float signed_disk_ring_coord(const SiPixelRecHit*);
  float signed_disk_ring_coord(const TrackingRecHit*);

  float blade_coord(const DetId&, const std::pair<int, int>&);
  float blade_coord(const DetId&, const PixelDigi*);
  float blade_coord(const DetId&, const SiPixelCluster*);
  float blade_coord(const SiPixelRecHit*);
  float blade_coord(const TrackingRecHit*);

  float signed_blade_coord(const DetId&, const std::pair<int, int>&);
  float signed_blade_coord(const DetId&, const PixelDigi*);
  float signed_blade_coord(const DetId&, const SiPixelCluster*);
  float signed_blade_coord(const SiPixelRecHit*);
  float signed_blade_coord(const TrackingRecHit*);

  float blade_panel_coord(const DetId&, const std::pair<int, int>&);
  float blade_panel_coord(const DetId&, const PixelDigi*);
  float blade_panel_coord(const DetId&, const SiPixelCluster*);
  float blade_panel_coord(const SiPixelRecHit*);
  float blade_panel_coord(const TrackingRecHit*);

  float signed_blade_panel_coord(const DetId&, const std::pair<int, int>&);
  float signed_blade_panel_coord(const DetId&, const PixelDigi*);
  float signed_blade_panel_coord(const DetId&, const SiPixelCluster*);
  float signed_blade_panel_coord(const SiPixelRecHit*);
  float signed_blade_panel_coord(const TrackingRecHit*);

  float signed_shifted_blade_panel_coord(const DetId&, const std::pair<int, int>&);
  float signed_shifted_blade_panel_coord(const DetId&, const PixelDigi*);
  float signed_shifted_blade_panel_coord(const DetId&, const SiPixelCluster*);
  float signed_shifted_blade_panel_coord(const SiPixelRecHit*);
  float signed_shifted_blade_panel_coord(const TrackingRecHit*);

private:
  int phase_;

  const TrackerTopology* tTopo_;
  const TrackerGeometry* tGeom_;
  const SiPixelFedCablingMap* cablingMap_;

  // Internal containers for optimal speed
  // - only calculate things once per DetId
  std::unordered_map<uint32_t, int> quadrant_;
  std::unordered_map<uint32_t, int> side_;
  std::unordered_map<uint32_t, int> module_;
  std::unordered_map<uint32_t, int> layer_;
  std::unordered_map<uint32_t, int> sector_;
  std::unordered_map<uint32_t, int> ladder_;
  std::unordered_map<uint32_t, int> signed_ladder_;
  std::unordered_map<uint32_t, int> signed_module_;
  std::unordered_map<uint32_t, int> half_;
  std::unordered_map<uint32_t, int> outer_;
  std::unordered_map<uint32_t, int> flipped_;
  std::unordered_map<uint32_t, int> disk_;
  std::unordered_map<uint32_t, int> signed_disk_;
  std::unordered_map<uint32_t, int> panel_;
  std::unordered_map<uint32_t, int> ring_;
  std::unordered_map<uint32_t, int> blade_;
  std::unordered_map<uint32_t, int> signed_blade_;

  std::unordered_map<uint32_t, unsigned int> fedid_;
  std::unordered_map<uint64_t, unsigned int> channel_;
  std::unordered_map<uint64_t, unsigned int> roc_;

  // Internal methods used for pixel coordinates
  bool isPixel_(const DetId&);
  bool isBPix_(const DetId&);
  bool isFPix_(const DetId&);
  std::pair<int, int> pixel_(const PixelDigi*);
  std::pair<int, int> pixel_(const SiPixelCluster*);
  std::pair<int, int> pixel_(const SiPixelRecHit*);
  float xcoord_on_module_(const DetId&, const std::pair<int, int>&);
  float ycoord_on_module_(const DetId&, const std::pair<int, int>&);
};

#endif
