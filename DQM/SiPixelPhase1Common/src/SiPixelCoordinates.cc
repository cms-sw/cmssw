// -*- C++ -*-
//
// Class:      SiPixelCoordinates
//
// Implementations of the class
//
// Original Author: Janos Karancsi

#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include <boost/range/irange.hpp>

// _________________________________________________________
//                 Constructors, destructor
SiPixelCoordinates::SiPixelCoordinates() { phase_ = -1; }

SiPixelCoordinates::SiPixelCoordinates(int phase) : phase_(phase) {}

SiPixelCoordinates::~SiPixelCoordinates() {}

// _________________________________________________________
//       init, called in the beginning of each event
void SiPixelCoordinates::init(const TrackerTopology* trackerTopology,
                              const TrackerGeometry* trackerGeometry,
                              const SiPixelFedCablingMap* siPixelFedCablingMap) {
  tTopo_ = trackerTopology;
  tGeom_ = trackerGeometry;
  cablingMap_ = siPixelFedCablingMap;

  fedid_ = cablingMap_->det2fedMap();

  // If not specified, determine from the geometry
  if (phase_ == -1) {
    if (tGeom_->isThere(GeomDetEnumerators::PixelBarrel) && tGeom_->isThere(GeomDetEnumerators::PixelEndcap))
      phase_ = 0;
    else if (tGeom_->isThere(GeomDetEnumerators::P1PXB) && tGeom_->isThere(GeomDetEnumerators::P1PXEC))
      phase_ = 1;
    else if (tGeom_->isThere(GeomDetEnumerators::P1PXB) && tGeom_->isThere(GeomDetEnumerators::P1PXEC))
      phase_ = 2;
  }
}

// _________________________________________________________
//       Offline/Online variables from TrackerTopology
//               and pixel naming classes

// Taken from pixel naming classes
// BmO (-z-x) = 1, BmI (-z+x) = 2 , BpO (+z-x) = 3 , BpI (+z+x) = 4
int SiPixelCoordinates::quadrant(const DetId& detid) {
  if (quadrant_.count(detid.rawId()))
    return quadrant_[detid.rawId()];
  if (!isPixel_(detid))
    return quadrant_[detid.rawId()] = -9999;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return quadrant_[detid.rawId()] = PixelBarrelName(detid, tTopo_, phase_).shell();
  else
    return quadrant_[detid.rawId()] = PixelEndcapName(detid, tTopo_, phase_).halfCylinder();
}

// Taken from Pixel naming class for barrel
// and TrackerTopology for endcap
// BmO/BmI = 1, BpO/BpI = 2
int SiPixelCoordinates::side(const DetId& detid) {
  if (side_.count(detid.rawId()))
    return side_[detid.rawId()];
  if (!isPixel_(detid))
    return side_[detid.rawId()] = -9999;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return side_[detid.rawId()] = 1 + (quadrant(detid) > 2);
  else
    return side_[detid.rawId()] = tTopo_->pxfSide(detid);
}

// Offline module convention taken from TrackerTopology
int SiPixelCoordinates::module(const DetId& detid) {
  if (module_.count(detid.rawId()))
    return module_[detid.rawId()];
  if (!isPixel_(detid))
    return module_[detid.rawId()] = -9999;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return module_[detid.rawId()] = tTopo_->pxbModule(detid.rawId());
  else
    return module_[detid.rawId()] = tTopo_->pxfModule(detid.rawId());
}

// Taken from TrackerTopology
int SiPixelCoordinates::layer(const DetId& detid) {
  if (layer_.count(detid.rawId()))
    return layer_[detid.rawId()];
  if (!isBPix_(detid))
    return layer_[detid.rawId()] = -9999;
  return layer_[detid.rawId()] = tTopo_->pxbLayer(detid);
}

// Taken from pixel naming class for barrel
int SiPixelCoordinates::sector(const DetId& detid) {
  if (sector_.count(detid.rawId()))
    return sector_[detid.rawId()];
  if (!isBPix_(detid))
    return sector_[detid.rawId()] = -9999;
  return sector_[detid.rawId()] = PixelBarrelName(detid, tTopo_, phase_).sectorName();
}

// Offline ladder convention taken from TrackerTopology
int SiPixelCoordinates::ladder(const DetId& detid) {
  if (ladder_.count(detid.rawId()))
    return ladder_[detid.rawId()];
  if (!isBPix_(detid))
    return ladder_[detid.rawId()] = -9999;
  return ladder_[detid.rawId()] = tTopo_->pxbLadder(detid);
}

// Online ladder convention taken from pixel naming class for barrel
// Apply sign convention (- sign for BmO and BpO)
int SiPixelCoordinates::signed_ladder(const DetId& detid) {
  if (signed_ladder_.count(detid.rawId()))
    return signed_ladder_[detid.rawId()];
  if (!isBPix_(detid))
    return signed_ladder_[detid.rawId()] = -9999;
  int signed_ladder = PixelBarrelName(detid, tTopo_, phase_).ladderName();
  if (quadrant(detid) % 2)
    signed_ladder *= -1;
  return signed_ladder_[detid.rawId()] = signed_ladder;
}

// Online mdoule convention taken from pixel naming class for barrel
// Apply sign convention (- sign for BmO and BmI)
int SiPixelCoordinates::signed_module(const DetId& detid) {
  if (signed_module_.count(detid.rawId()))
    return signed_module_[detid.rawId()];
  if (!isBPix_(detid))
    return signed_module_[detid.rawId()] = -9999;
  int signed_module = PixelBarrelName(detid, tTopo_, phase_).moduleName();
  if (quadrant(detid) < 3)
    signed_module *= -1;
  return signed_module_[detid.rawId()] = signed_module;
}

// Half ladders taken from pixel naming class
int SiPixelCoordinates::half(const DetId& detid) {
  if (half_.count(detid.rawId()))
    return half_[detid.rawId()];
  if (!isBPix_(detid))
    return half_[detid.rawId()] = -9999;
  return half_[detid.rawId()] = PixelBarrelName(detid, tTopo_, phase_).isHalfModule();
}

// Using TrackerTopology
// Ladders have a staggered structure
// Non-flipped ladders are on the outer radius
// Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
// Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
int SiPixelCoordinates::outer(const DetId& detid) {
  if (outer_.count(detid.rawId()))
    return outer_[detid.rawId()];
  if (!isBPix_(detid))
    return outer_[detid.rawId()] = -9999;
  int outer = -9999;
  int layer = tTopo_->pxbLayer(detid.rawId());
  bool odd_ladder = tTopo_->pxbLadder(detid.rawId()) % 2;
  if (phase_ == 0) {
    if (layer == 2)
      outer = !odd_ladder;
    else
      outer = odd_ladder;
  } else if (phase_ == 1) {
    if (layer == 4)
      outer = odd_ladder;
    else
      outer = !odd_ladder;
  }
  return outer_[detid.rawId()] = outer;
}

// Using outer() method
// We call ladders in the inner radius flipped (see above)
int SiPixelCoordinates::flipped(const DetId& detid) {
  if (flipped_.count(detid.rawId()))
    return flipped_[detid.rawId()];
  if (!isBPix_(detid))
    return flipped_[detid.rawId()] = -9999;
  int flipped = -9999;
  if (phase_ < 2)
    flipped = outer(detid) == 0;
  return flipped_[detid.rawId()] = flipped;
}

// Offline disk convention taken from TrackerTopology
int SiPixelCoordinates::disk(const DetId& detid) {
  if (disk_.count(detid.rawId()))
    return disk_[detid.rawId()];
  if (!isFPix_(detid))
    return disk_[detid.rawId()] = -9999;
  return disk_[detid.rawId()] = tTopo_->pxfDisk(detid);
}

// Online disk convention
// Apply sign convention (- sign for BmO and BmI)
int SiPixelCoordinates::signed_disk(const DetId& detid) {
  if (signed_disk_.count(detid.rawId()))
    return signed_disk_[detid.rawId()];
  if (!isFPix_(detid))
    return signed_disk_[detid.rawId()] = -9999;
  int signed_disk = disk(detid);
  if (quadrant(detid) < 3)
    signed_disk *= -1;
  return signed_disk_[detid.rawId()] = signed_disk;
}

// Taken from TrackerTopology
int SiPixelCoordinates::panel(const DetId& detid) {
  if (panel_.count(detid.rawId()))
    return panel_[detid.rawId()];
  if (!isFPix_(detid))
    return panel_[detid.rawId()] = -9999;
  return panel_[detid.rawId()] = tTopo_->pxfPanel(detid);
}

// Phase 0: Ring was not an existing convention
//   but the 7 plaquettes were split by HV group
//   --> Derive Ring 1/2 for them
//   Panel 1 plq 1-2, Panel 2, plq 1   = Ring 1
//   Panel 1 plq 3-4, Panel 2, plq 2-3 = Ring 2
// Phase 1: Using pixel naming class for endcap
int SiPixelCoordinates::ring(const DetId& detid) {
  if (ring_.count(detid.rawId()))
    return ring_[detid.rawId()];
  if (!isFPix_(detid))
    return ring_[detid.rawId()] = -9999;
  int ring = -9999;
  if (phase_ == 0) {
    ring = 1 + (panel(detid) + module(detid) > 3);
  } else if (phase_ == 1) {
    ring = PixelEndcapName(detid, tTopo_, phase_).ringName();
  }
  return ring_[detid.rawId()] = ring;
}

// Offline blade convention taken from TrackerTopology
int SiPixelCoordinates::blade(const DetId& detid) {
  if (blade_.count(detid.rawId()))
    return blade_[detid.rawId()];
  if (!isFPix_(detid))
    return blade_[detid.rawId()] = -9999;
  return blade_[detid.rawId()] = tTopo_->pxfBlade(detid);
}

// Online blade convention taken from pixel naming class for endcap
// Apply sign convention (- sign for BmO and BpO)
int SiPixelCoordinates::signed_blade(const DetId& detid) {
  if (signed_blade_.count(detid.rawId()))
    return signed_blade_[detid.rawId()];
  if (!isFPix_(detid))
    return signed_blade_[detid.rawId()] = -9999;
  int signed_blade = PixelEndcapName(detid, tTopo_, phase_).bladeName();
  if (quadrant(detid) % 2)
    signed_blade *= -1;
  return signed_blade_[detid.rawId()] = signed_blade;
}

// Get the FED number using the cabling map
unsigned int SiPixelCoordinates::fedid(const DetId& detid) {
  if (fedid_.count(detid.rawId()))
    return fedid_[detid.rawId()];
  if (!isPixel_(detid))
    return fedid_[detid.rawId()] = 9999;
  unsigned int fedid = 9999;
  for (auto& fedId : cablingMap_->fedIds()) {
    if (SiPixelFrameConverter(cablingMap_, fedId).hasDetUnit(detid.rawId())) {
      fedid = fedId;
      break;
    }
  }
  return fedid_[detid.rawId()] = fedid;
}

// _________________________________________________________
//                    Private methods
bool SiPixelCoordinates::isPixel_(const DetId& detid) {
  if (detid.det() != DetId::Tracker)
    return false;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return true;
  if (detid.subdetId() == PixelSubdetector::PixelEndcap)
    return true;
  return false;
}
bool SiPixelCoordinates::isBPix_(const DetId& detid) {
  if (detid.det() != DetId::Tracker)
    return false;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return true;
  return false;
}
bool SiPixelCoordinates::isFPix_(const DetId& detid) {
  if (detid.det() != DetId::Tracker)
    return false;
  if (detid.subdetId() == PixelSubdetector::PixelEndcap)
    return true;
  return false;
}

std::pair<int, int> SiPixelCoordinates::pixel_(const PixelDigi* digi) {
  return std::make_pair(digi->row(), digi->column());
}
std::pair<int, int> SiPixelCoordinates::pixel_(const SiPixelCluster* cluster) {
  // Cluster positions are already shifted by 0.5
  // We remove this and add back later (for all pixels)
  // The aim is to get the offline row/col number of the pixel
  int row = cluster->x() - 0.5, col = cluster->y() - 0.5;
  return std::make_pair(row, col);
}
std::pair<int, int> SiPixelCoordinates::pixel_(const SiPixelRecHit* rechit) {
  // Convert RecHit local position to local pixel using Topology
  const PixelGeomDetUnit* detUnit = static_cast<const PixelGeomDetUnit*>(rechit->detUnit());
  const PixelTopology* topo = static_cast<const PixelTopology*>(&detUnit->specificTopology());
  std::pair<float, float> pixel = topo->pixel(rechit->localPosition());
  // We could leave it like this, but it's better to constrain pixel to be on the module
  // Also truncate floating point to int (similar to digis)
  int row = std::max(0, std::min(topo->nrows() - 1, (int)pixel.first));
  int col = std::max(0, std::min(topo->ncolumns() - 1, (int)pixel.second));
  return std::make_pair(row, col);
}

float SiPixelCoordinates::xcoord_on_module_(const DetId& detid, const std::pair<int, int>& pixel) {
  int nrows = 160;
  // Leave it hard-coded for phase 0/1, read from geometry for phase 2
  // no special treatment needed here for phase 0 1x8, 1x5 and 1x2 modules either
  // because we do not want to scale coordinates (only shift if needed)
  if (phase_ == 2) {
    const PixelGeomDetUnit* detUnit = static_cast<const PixelGeomDetUnit*>(tGeom_->idToDetUnit(detid));
    const PixelTopology* topo = static_cast<const PixelTopology*>(&detUnit->specificTopology());
    nrows = topo->nrows();
  }
  // Shift to the middle of the pixel, for precision binning
  return (pixel.first + 0.5) / nrows;
}

float SiPixelCoordinates::ycoord_on_module_(const DetId& detid, const std::pair<int, int>& pixel) {
  int ncols = 416;
  // Leave it hard-coded for phase 0/1, read from geometry for phase 2
  if (phase_ == 2) {
    const PixelGeomDetUnit* detUnit = static_cast<const PixelGeomDetUnit*>(tGeom_->idToDetUnit(detid));
    const PixelTopology* topo = static_cast<const PixelTopology*>(&detUnit->specificTopology());
    ncols = topo->ncolumns();
  } else if (phase_ == 0 && isFPix_(detid)) {
    // Always use largest length for Phase 0 FPix modules (1x5 and 2x5)
    // because we do not want to scale coordinates so ROC size remains fixed
    // and only shifts are needed
    ncols = 260;
  }
  // Shift to the middle of the pixel, for precision binning
  return (pixel.second + 0.5) / ncols;
}

// _________________________________________________________
//                 Online Link and ROC number

// Get the FED channel (link) number
// Link may depend on the TBM side of the module
// so pixel location is needed
// Using the cabling map works for all detectors
// Taken from DQM/SiPixelMonitorClient/src/SiPixelInformationExtractor.cc
int SiPixelCoordinates::channel(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isPixel_(detid))
    return -9999;
  // The method below may be slow when looping on a lot of pixels, so let's try to speed it up
  // by quickly chategorizing pixels to ROC coordinates inside det units
  int rowsperroc = 80, colsperroc = 52;
  if (phase_ == 2) {
    // Can get roc info from Geometry for Phase 2, this will need to be specified when it's final
    const PixelGeomDetUnit* detUnit = static_cast<const PixelGeomDetUnit*>(tGeom_->idToDetUnit(detid));
    const PixelTopology* topo = static_cast<const PixelTopology*>(&detUnit->specificTopology());
    rowsperroc = topo->rowsperroc();
    colsperroc = topo->colsperroc();
  }
  // It is unlikely a ROC would have more than 256 chips, so let's use this formula
  // If a ROC number was ever found, then binary search in a map will be much quicker
  uint64_t pseudo_roc_num =
      uint64_t(1 << 16) * detid.rawId() + (1 << 8) * (pixel.first / rowsperroc) + pixel.second / colsperroc;
  if (channel_.count(pseudo_roc_num))
    return channel_[pseudo_roc_num];
  // If not found previously, get the channel number
  unsigned int fedId = fedid(detid);
  SiPixelFrameConverter converter(cablingMap_, fedId);
  sipixelobjects::DetectorIndex detector = {detid.rawId(), pixel.first, pixel.second};
  sipixelobjects::ElectronicIndex cabling;
  converter.toCabling(cabling, detector);
  // Time consuming part is over, so let's save the roc number too
  const sipixelobjects::PixelROC* theRoc = converter.toRoc(cabling.link, cabling.roc);
  int roc = theRoc->idInDetUnit();
  if (detid.subdetId() == PixelSubdetector::PixelBarrel && side(detid) == 1 && half(detid))
    roc += 8;
  roc_[pseudo_roc_num] = roc;
  //printf ("Online FED, LNK, LNKID, ROC: %2d %2d %2d %2d - Offline RAWID, ROW, COL: %9d [%3d,%3d] [%3d,%3d]\n",
  //        fedId, cabling.link, cabling.roc, roc, detid.rawId(),
  //        (pixel.first /rowsperroc)*rowsperroc, (pixel.first /rowsperroc+1)*rowsperroc-1,
  //        (pixel.second/colsperroc)*colsperroc, (pixel.second/colsperroc+1)*colsperroc-1);
  return channel_[pseudo_roc_num] = cabling.link;
}
int SiPixelCoordinates::channel(const DetId& detid, const PixelDigi* digi) {
  if (!isPixel_(detid))
    return -9999;
  return channel(detid, pixel_(digi));
}
int SiPixelCoordinates::channel(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isPixel_(detid))
    return -9999;
  return channel(detid, pixel_(cluster));
}
int SiPixelCoordinates::channel(const SiPixelRecHit* rechit) {
  if (!isPixel_(rechit->geographicalId()))
    return -9999;
  return channel(rechit->geographicalId(), pixel_(rechit));
}
int SiPixelCoordinates::channel(const TrackingRecHit* rechit) {
  if (!isPixel_(rechit->geographicalId()))
    return -9999;
  return channel(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Using the cabling map works for all detectors
// Taken from DQM/SiPixelMonitorClient/src/SiPixelInformationExtractor.cc
// Although using coordinates (only available for Phase 0/1) is much faster
// The advantage is very visible when running on smaller statistics
// because the map will speed it up greatly after high enough ROCs were sampled
// The coordinate method is validated to give the same result as the cabling map
// Example for the barrel:
// ROC number is read out in a U shape from ROC 0 to 15 (or maxroc)
// row [80-159] col [0-51] is always ROC 0 on the +Z side of the barrel
// Both coordinates are mirrored on the -Z side (180 deg rotation effectively)
// -Z        8  9 10 11 12 13 14 15   +Z        0  1  2  3  4  5  6  7
//    (0,0)  7  6  5  4  3  2  1  0      (0,0) 15 14 13 12 11 10  9  8
// Half modules on the -Z side should consider the second row of ROCs instead, etc. see below
int SiPixelCoordinates::roc(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isPixel_(detid))
    return -9999;
  // The method below may be slow when looping on a lot of pixels, so let's try to speed it up
  // by quickly chategorizing pixels to ROC coordinates inside det units
  int rowsperroc = 80, colsperroc = 52;
  if (phase_ == 2) {
    // Can get roc info from Geometry for Phase 2, this will need to be specified when it's final
    const PixelGeomDetUnit* detUnit = static_cast<const PixelGeomDetUnit*>(tGeom_->idToDetUnit(detid));
    const PixelTopology* topo = static_cast<const PixelTopology*>(&detUnit->specificTopology());
    rowsperroc = topo->rowsperroc();
    colsperroc = topo->colsperroc();
  }
  // It is unlikely a ROC would have more than 256 chips, so let's use this formula
  // If a ROC number was ever found, then binary search in a map will be much quicker
  uint64_t pseudo_roc_num =
      uint64_t(1 << 16) * detid.rawId() + (1 << 8) * (pixel.first / rowsperroc) + pixel.second / colsperroc;
  if (roc_.count(pseudo_roc_num))
    return roc_[pseudo_roc_num];
  // If not found previously, get the ROC number
  int roc = -9999;
  // Use the Fed Cabling Map if specified by the bool
  // or if using channel number too, or if it's the Phase 2 detector
  if (phase_ == 2 || !channel_.empty()) {
    unsigned int fedId = fedid(detid);
    SiPixelFrameConverter converter(cablingMap_, fedId);
    sipixelobjects::DetectorIndex detector = {detid.rawId(), pixel.first, pixel.second};
    sipixelobjects::ElectronicIndex cabling;
    converter.toCabling(cabling, detector);
    // Time consuming part is over, so let's save the channel number too
    channel_[pseudo_roc_num] = cabling.link;
    const sipixelobjects::PixelROC* theRoc = converter.toRoc(cabling.link, cabling.roc);
    roc = theRoc->idInDetUnit();
    if (detid.subdetId() == PixelSubdetector::PixelBarrel && side(detid) == 1 && half(detid))
      roc += 8;
    //printf ("Online FED, LNK, LNKID, ROC: %2d %2d %2d %2d - Offline RAWID, ROW, COL: %9d [%3d,%3d] [%3d,%3d]\n",
    //        fedId, cabling.link, cabling.roc, roc, detid.rawId(),
    //        (pixel.first /rowsperroc)*rowsperroc, (pixel.first /rowsperroc+1)*rowsperroc-1,
    //        (pixel.second/colsperroc)*colsperroc, (pixel.second/colsperroc+1)*colsperroc-1);
  } else if (phase_ < 2) {
    // This method is faster if only ROC number is needed
    int pan = panel(detid), mod = module(detid), rocsY = 8;
    if (phase_ == 0 && detid.subdetId() == PixelSubdetector::PixelEndcap)
      rocsY = pan + mod;
    int rocX = pixel.first / rowsperroc, rocY = pixel.second / colsperroc;
    // Consider second row for all 1xN Phase 0 modules
    if (phase_ == 0) {
      int v1x8 = half(detid) == 1, v1x2 = (pan == 1 && mod == 1), v1x5 = (pan == 1 && mod == 4);
      if (v1x8 || v1x2 || v1x5)
        ++rocX;
    }
    // Mirror both coordinates for barrel -Z side
    // and for endcap (but only Panel 2 for Phase 0)
    if ((detid.subdetId() == PixelSubdetector::PixelBarrel && side(detid) == 1) ||
        (detid.subdetId() == PixelSubdetector::PixelEndcap && ((phase_ == 0 && pan == 2) || phase_ == 1))) {
      rocX = 1 - rocX;
      rocY = rocsY - 1 - rocY;
    }
    // U-shape readout order
    roc = rocX ? rocY : 2 * rocsY - 1 - rocY;
  }
  return roc_[pseudo_roc_num] = roc;
}
int SiPixelCoordinates::roc(const DetId& detid, const PixelDigi* digi) {
  if (!isPixel_(detid))
    return -9999;
  return roc(detid, pixel_(digi));
}
int SiPixelCoordinates::roc(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isPixel_(detid))
    return -9999;
  return roc(detid, pixel_(cluster));
}
int SiPixelCoordinates::roc(const SiPixelRecHit* rechit) {
  if (!isPixel_(rechit->geographicalId()))
    return -9999;
  return roc(rechit->geographicalId(), pixel_(rechit));
}
int SiPixelCoordinates::roc(const TrackingRecHit* rechit) {
  if (!isPixel_(rechit->geographicalId()))
    return -9999;
  return roc(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// _________________________________________________________
//    Floating point Pixel Coordinates similar to those
//       given by TrackerTopology and naming classes
//          but we add a shift within ]-0.5,+0.5[
//    eg. std::round(coord) gives back the original int
float SiPixelCoordinates::module_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isBPix_(detid))
    return -9999;
  // offline module number is monotonously increasing with global z
  // sign is negative because local y is antiparallel to global z
  return module(detid) - (ycoord_on_module_(detid, pixel) - 0.5);
}
float SiPixelCoordinates::module_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isBPix_(detid))
    return -9999;
  return module_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::module_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isBPix_(detid))
    return -9999;
  return module_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::module_coord(const SiPixelRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return module_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::module_coord(const TrackingRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return module_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

float SiPixelCoordinates::signed_module_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isBPix_(detid))
    return -9999;
  // offline module number is monotonously increasing with global z
  // sign is negative because local y is antiparallel to global z
  return signed_module(detid) - (ycoord_on_module_(detid, pixel) - 0.5);
}
float SiPixelCoordinates::signed_module_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isBPix_(detid))
    return -9999;
  return signed_module_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_module_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isBPix_(detid))
    return -9999;
  return signed_module_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_module_coord(const SiPixelRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return signed_module_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_module_coord(const TrackingRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return signed_module_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

float SiPixelCoordinates::ladder_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isBPix_(detid))
    return -9999;
  // offline ladder number is monotonously increasing with global phi
  // flipped/inner ladders:     lx parallel to global r-phi - positive sign
  // non-flipped/outer ladders: lx anti-parallel to global r-phi - negative sign
  int sign = flipped(detid) ? 1 : -1;
  return ladder(detid) + sign * (xcoord_on_module_(detid, pixel) + half(detid) * 0.5 - 0.5);
}
float SiPixelCoordinates::ladder_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isBPix_(detid))
    return -9999;
  return ladder_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::ladder_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isBPix_(detid))
    return -9999;
  return ladder_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::ladder_coord(const SiPixelRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return ladder_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::ladder_coord(const TrackingRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return ladder_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

float SiPixelCoordinates::signed_ladder_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isBPix_(detid))
    return -9999;
  // online ladder number is monotonously decreasing with global phi
  // flipped/inner ladders:     lx parallel to global r-phi - negative sign
  // non-flipped/outer ladders: lx anti-parallel to global r-phi - positive sign
  int sign = flipped(detid) ? -1 : 1;
  return signed_ladder(detid) + sign * (xcoord_on_module_(detid, pixel) + half(detid) * 0.5 - 0.5);
}
float SiPixelCoordinates::signed_ladder_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isBPix_(detid))
    return -9999;
  return signed_ladder_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_ladder_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isBPix_(detid))
    return -9999;
  return signed_ladder_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_ladder_coord(const SiPixelRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return signed_ladder_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_ladder_coord(const TrackingRecHit* rechit) {
  if (!isBPix_(rechit->geographicalId()))
    return -9999;
  return signed_ladder_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Rings are defined in the radial direction
// which is local x for phase 0 and local y for phase 1
// Rings were not defined for phase 0, but we had a similar
// convention, HV group, the 7 plaquettes were split like this
//   Panel 1 plq 1-2, Panel 2, plq 1   = Ring 1 (HV grp 1)
//   Panel 1 plq 3-4, Panel 2, plq 2-3 = Ring 2 (HV grp 2)
// A subdivision of 8 is suggested for both phase 0 and 1
float SiPixelCoordinates::ring_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float ring_coord = ring(detid), coord_shift = 0;
  if (phase_ == 0) {
    // local x on panel 1 is anti-parallel to global radius - sign is negative
    // and parallel for panel 2 - sign is positive
    int pan = panel(detid), mod = module(detid);
    if (pan == 1) {
      if (mod == 1)
        coord_shift = (-xcoord_on_module_(detid, pixel)) / 4;
      else if (mod == 2)
        coord_shift = (-xcoord_on_module_(detid, pixel) + 2.0) / 4;
      else if (mod == 3)
        coord_shift = (-xcoord_on_module_(detid, pixel)) / 4;
      else if (mod == 4)
        coord_shift = (-xcoord_on_module_(detid, pixel) + 1.5) / 4;
    } else {
      if (mod == 1)
        coord_shift = (xcoord_on_module_(detid, pixel)) / 4;
      else if (mod == 2)
        coord_shift = (xcoord_on_module_(detid, pixel) - 2.0) / 4;
      else if (mod == 3)
        coord_shift = (xcoord_on_module_(detid, pixel)) / 4;
    }
  } else if (phase_ == 1) {
    // local y is parallel to global radius, so sign is positive
    coord_shift = ycoord_on_module_(detid, pixel) - 0.5;
  }
  ring_coord += coord_shift;
  return ring_coord;
}
float SiPixelCoordinates::ring_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return ring_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::ring_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return ring_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::ring_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return ring_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::ring_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return ring_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Treat disk number as it is (parallel to global z)
// Subdivisions on the forward can be the radial direction
// Which is local x for phase 0 and local y for phase 1
// Closest radius is chosen to be closest to disk = 0
// Rings are not separated, 8 subdivisions are suggested
// Plot suitable for separate ring plots
float SiPixelCoordinates::disk_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float disk_coord = disk(detid), coord_shift = ring_coord(detid, pixel) - ring(detid);
  disk_coord += coord_shift;
  return disk_coord;
}
float SiPixelCoordinates::disk_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return disk_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::disk_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return disk_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::disk_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return disk_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::disk_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return disk_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Same as above, but using online convention
// !!! Recommended for Phase 1 !!!
// Can be used for Phase 0 too for comparison purposes
float SiPixelCoordinates::signed_disk_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float signed_disk_coord = signed_disk(detid), coord_shift = ring_coord(detid, pixel) - ring(detid);
  // Mirror -z side, so plots are symmetric
  if (signed_disk_coord < 0)
    coord_shift = -coord_shift;
  signed_disk_coord += coord_shift;
  return signed_disk_coord;
}
float SiPixelCoordinates::signed_disk_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return signed_disk_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_disk_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return signed_disk_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_disk_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_disk_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_disk_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_disk_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Same as the above two, but subdivisions incorporate rings as well
// 16 subdivisions are suggested
float SiPixelCoordinates::disk_ring_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float disk_ring_coord = disk(detid), coord_shift = 0;
  //if      (phase_==0) coord_shift = (ring_coord(detid,pixel) - 1.625) / 1.5;
  //else if (phase_==1) coord_shift = (ring_coord(detid,pixel) - 1.5  ) / 2.0;
  coord_shift = (ring_coord(detid, pixel) - 1.5) / 2.0;
  disk_ring_coord += coord_shift;
  return disk_ring_coord;
}
float SiPixelCoordinates::disk_ring_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return disk_ring_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::disk_ring_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return disk_ring_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::disk_ring_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return disk_ring_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::disk_ring_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return disk_ring_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Same as above, but using online convention
// !!! Recommended for Phase 0 !!!
float SiPixelCoordinates::signed_disk_ring_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float signed_disk_ring_coord = signed_disk(detid), coord_shift = 0;
  //if      (phase_==0) coord_shift = (ring_coord(detid,pixel) - 1.625) / 1.5;
  //else if (phase_==1) coord_shift = (ring_coord(detid,pixel) - 1.5  ) / 2.0;
  coord_shift = (ring_coord(detid, pixel) - 1.5) / 2.0;
  // Mirror -z side, so plots are symmetric
  if (signed_disk_ring_coord < 0)
    coord_shift = -coord_shift;
  signed_disk_ring_coord += coord_shift;
  return signed_disk_ring_coord;
}
float SiPixelCoordinates::signed_disk_ring_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return signed_disk_ring_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_disk_ring_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return signed_disk_ring_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_disk_ring_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_disk_ring_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_disk_ring_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_disk_ring_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Offline blade convention
// Blade number is parallel to global phi
// For Phase 0: local y is parallel with phi
//   On +Z side ly is parallel with phi
//   On -Z side ly is anti-parallel
// Phase 1: local x is parallel with phi
//   +Z Panel 1, -Z Panel 2 is parallel
//   +Z Panel 2, -Z Panel 1 is anti-parallel
// Plot suitable for separate panel 1/2 plots
// 10 subdivisions are recommended for Phase 0 (Half-ROC granularity)
// 2 for Phase 1
float SiPixelCoordinates::blade_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float blade_coord = blade(detid), coord_shift = 0;
  if (phase_ == 0) {
    int rocsY = panel(detid) + module(detid);
    coord_shift = ycoord_on_module_(detid, pixel) - rocsY / 10.;
    if (side(detid) == 1)
      coord_shift = -coord_shift;
  } else if (phase_ == 1) {
    coord_shift = xcoord_on_module_(detid, pixel) - 0.5;
    if ((side(detid) + panel(detid)) % 2 == 0)
      coord_shift = -coord_shift;
  }
  blade_coord += coord_shift;
  return blade_coord;
}
float SiPixelCoordinates::blade_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return blade_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::blade_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return blade_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::blade_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return blade_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::blade_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return blade_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Online blade convention
// Blade number is anti-parallel to global phi
// so signs are the opposite as above
// Plot suitable for separate panel 1/2 plots
// 10 subdivisions are recommended for Phase 0 (Half-ROC granularity)
// 2 for Phase 1
// !!! Recommended for Phase 0 |||
float SiPixelCoordinates::signed_blade_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float signed_blade_coord = signed_blade(detid), coord_shift = 0;
  if (phase_ == 0) {
    int rocsY = panel(detid) + module(detid);
    coord_shift = ycoord_on_module_(detid, pixel) - rocsY / 10.;
    if (side(detid) == 2)
      coord_shift = -coord_shift;
  } else if (phase_ == 1) {
    coord_shift = xcoord_on_module_(detid, pixel) - 0.5;
    if ((side(detid) + panel(detid)) % 2 == 1)
      coord_shift = -coord_shift;
  }
  signed_blade_coord += coord_shift;
  return signed_blade_coord;
}
float SiPixelCoordinates::signed_blade_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return signed_blade_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_blade_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return signed_blade_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_blade_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_blade_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_blade_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_blade_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Offline blade convention + alternating panels
// Same as above two, but subdivisions incorporate panels
// Panel 2 is towards higher phi values for Phase 1 (overlap for phase 0)
// 20 subdivisions are recommended for Phase 0 (Half-ROC granularity)
// 4 for Phase 1
float SiPixelCoordinates::blade_panel_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float blade_panel_coord = blade(detid);
  float coord_shift = (blade_coord(detid, pixel) - blade_panel_coord + panel(detid) - 1.5) / 2;
  blade_panel_coord += coord_shift;
  return blade_panel_coord;
}
float SiPixelCoordinates::blade_panel_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return blade_panel_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::blade_panel_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return blade_panel_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::blade_panel_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return blade_panel_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::blade_panel_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return blade_panel_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Online blade convention + alternating panels
// Blade number is anti-parallel to global phi
// so signs are the opposite as above
// 20 subdivisions are recommended for Phase 0 (Half-ROC granularity)
// 4 for Phase 1
// !!! Recommended for Phase 1 !!!
float SiPixelCoordinates::signed_blade_panel_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float signed_blade_panel_coord = signed_blade(detid);
  float coord_shift = (signed_blade_coord(detid, pixel) - signed_blade_panel_coord - panel(detid) + 1.5) / 2;
  signed_blade_panel_coord += coord_shift;
  return signed_blade_panel_coord;
}
float SiPixelCoordinates::signed_blade_panel_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return signed_blade_panel_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_blade_panel_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return signed_blade_panel_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_blade_panel_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_blade_panel_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_blade_panel_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_blade_panel_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}

// Same as above, but blade numbers are shifted for Phase 1 Ring 1
// so one can plot Ring1+Ring2 while conserving geometrical
// overlaps in phi
// Ring 2: 17 blades x 4 ROC --> 68 bin
// Ring 1: 2 gap, 4 ROC, alternating for 11 blades --> 68 bin
float SiPixelCoordinates::signed_shifted_blade_panel_coord(const DetId& detid, const std::pair<int, int>& pixel) {
  if (!isFPix_(detid))
    return -9999;
  float signed_shifted_blade_panel_coord = signed_blade(detid);
  float coord_shift = (signed_blade_coord(detid, pixel) - signed_shifted_blade_panel_coord - panel(detid) + 1.5) / 2;
  if (phase_ == 1 && ring(detid) == 1)
    signed_shifted_blade_panel_coord *= 1.5;
  signed_shifted_blade_panel_coord += coord_shift;
  return signed_shifted_blade_panel_coord;
}
float SiPixelCoordinates::signed_shifted_blade_panel_coord(const DetId& detid, const PixelDigi* digi) {
  if (!isFPix_(detid))
    return -9999;
  return signed_shifted_blade_panel_coord(detid, pixel_(digi));
}
float SiPixelCoordinates::signed_shifted_blade_panel_coord(const DetId& detid, const SiPixelCluster* cluster) {
  if (!isFPix_(detid))
    return -9999;
  return signed_shifted_blade_panel_coord(detid, pixel_(cluster));
}
float SiPixelCoordinates::signed_shifted_blade_panel_coord(const SiPixelRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_shifted_blade_panel_coord(rechit->geographicalId(), pixel_(rechit));
}
float SiPixelCoordinates::signed_shifted_blade_panel_coord(const TrackingRecHit* rechit) {
  if (!isFPix_(rechit->geographicalId()))
    return -9999;
  return signed_shifted_blade_panel_coord(static_cast<const SiPixelRecHit*>(rechit->hit()));
}
