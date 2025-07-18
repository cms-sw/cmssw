#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "Geometry/HGCalCommonData/interface/HGCGuardRing.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCGuardRing::HGCGuardRing(const HGCalDDDConstants& hgc)
    : hgcons_(hgc),
      modeUV_(hgcons_.geomMode()),
      v17OrLess_(hgcons_.v17OrLess()),
      waferSize_(hgcons_.waferSize(false)),
      sensorSizeOffset_(hgcons_.sensorSizeOffset(false)),
      guardRingOffset_(hgcons_.guardRingOffset(false)) {
  offset_ = sensorSizeOffset_ + guardRingOffset_;
  offsetPartial_ = guardRingOffset_;
  xmax_ = 0.5 * waferSize_ - offset_;
  ymax_ = xmax_ / sqrt3_;
  c22_ = (v17OrLess_) ? HGCalTypes::c22O : HGCalTypes::c22;
  c27_ = (v17OrLess_) ? HGCalTypes::c27O : HGCalTypes::c27;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCGuardRing with wafer size " << waferSize_ << ", Offsets "
                             << sensorSizeOffset_ << ":" << guardRingOffset_ << ":" << offset_ << ":" << offsetPartial_
                             << ", mode " << modeUV_ << ", xmax|ymax " << xmax_ << ":" << ymax_ << " and c22:c27 "
                             << c22_ << ":" << c27_;
#endif
}

bool HGCGuardRing::exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV) {
  bool check(false);
  if (hgcons_.waferHexagon8Module()) {
    int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV);
    int partial = HGCalWaferType::getPartial(index, hgcons_.getParameter()->waferInfoMap_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCGuardRing::exclude: Layer " << layer << " wafer " << waferU << ":" << waferV
                               << " index " << index << " partial " << partial;
#endif
    if (partial == HGCalTypes::WaferFull) {
      double dx = std::abs(point.x());
      double dy = std::abs(point.y());
      if (dx > xmax_) {
        check = true;
      } else if (dy > (2 * ymax_)) {
        check = true;
      } else {
        check = (dx > (sqrt3_ * (2 * ymax_ - dy)));
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "HGCGuardRing::exclude: Point " << point << " zside " << zside << " layer " << layer
                                 << " wafer " << waferU << ":" << waferV << " partial type " << partial << ":"
                                 << HGCalTypes::WaferFull << " x " << dx << ":" << xmax_ << " y " << dy << ":" << ymax_
                                 << " check " << check;
#endif
    } else if (partial > 0) {
      int orient = HGCalWaferType::getOrient(index, hgcons_.getParameter()->waferInfoMap_);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "HGCGuardRing::exclude: Orient " << orient << " Mode " << modeUV_;
#endif
      if (hgcons_.v16OrLess()) {
        std::vector<std::pair<double, double> > wxy =
            HGCalWaferMask::waferXY(partial, orient, zside, waferSize_, offset_, 0.0, 0.0, v17OrLess_);
        check = !(insidePolygon(point.x(), point.y(), wxy));
#ifdef EDM_ML_DEBUG
        std::ostringstream st1;
        st1 << "HGCGuardRing::exclude: Point " << point << " Partial/orient/zside/size/offset " << partial << ":"
            << orient << ":" << zside << ":" << waferSize_ << offset_ << " with " << wxy.size() << " points:";
        for (unsigned int k = 0; k < wxy.size(); ++k)
          st1 << " (" << wxy[k].first << ", " << wxy[k].second << ")";
        edm::LogVerbatim("HGCSim") << st1.str();
#endif
      } else {
        int placement = HGCalCell::cellPlacementIndex(zside, frontBack, orient);
        std::vector<std::pair<double, double> > wxy =
            HGCalWaferMask::waferXY(partial, placement, waferSize_, offset_, 0.0, 0.0, v17OrLess_);
        check = !(insidePolygon(point.x(), point.y(), wxy));
#ifdef EDM_ML_DEBUG
        std::ostringstream st1;
        st1 << "HGCGuardRing::exclude: Point " << point << " Partial/frontback/orient/zside/placeemnt/size/offset "
            << partial << ":" << frontBack << ":" << orient << ":" << zside << ":" << placement << ":" << waferSize_
            << offset_ << " with " << wxy.size() << " points:";
        for (unsigned int k = 0; k < wxy.size(); ++k)
          st1 << " (" << wxy[k].first << ", " << wxy[k].second << ")";
        edm::LogVerbatim("HGCSim") << st1.str();
#endif
      }
    } else {
      check = true;
    }
  }
  return check;
}

bool HGCGuardRing::excludePartial(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV) {
  bool check(false);
  if (hgcons_.waferHexagon8Cassette()) {
    int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV);
    int partial = HGCalWaferType::getPartial(index, hgcons_.getParameter()->waferInfoMap_);
    int type = HGCalWaferType::getType(index, hgcons_.getParameter()->waferInfoMap_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCGuardRing::excludePartial: Layer " << layer << " wafer " << waferU << ":"
                               << waferV << " index " << index << " partial " << partial << " type " << type;
#endif
    if (partial == HGCalTypes::WaferFull) {
      return (check);
    } else if (partial < 0) {
      return true;
    } else {
      int orient = HGCalWaferType::getOrient(index, hgcons_.getParameter()->waferInfoMap_);
      int placement = HGCalCell::cellPlacementIndex(zside, frontBack, orient);
      double dx = point.x();
      double dy = point.y();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "HGCGuardRing::excludePartial: zside " << zside << " frontBack " << frontBack
                                 << " orient " << orient << " placement " << placement << " dx " << dx << " dy " << dy;
#endif
      if (type > 0) {
        for (int ii = HGCalTypes::WaferPartLDOffset;
             ii < (HGCalTypes::WaferPartLDOffset + HGCalTypes::WaferPartLDCount);
             ii++) {
          std::array<double, 4> criterion =
              HGCalWaferMask::maskCut(ii, placement, waferSize_, offsetPartial_, v17OrLess_);
          check |= std::abs(criterion[0] * dy + criterion[1] * dx + criterion[2]) < criterion[3];
        }
      } else {
        for (int ii = HGCalTypes::WaferPartHDOffset;
             ii < (HGCalTypes::WaferPartHDOffset + HGCalTypes::WaferPartHDCount);
             ii++) {
          std::array<double, 4> criterion =
              HGCalWaferMask::maskCut(ii, placement, waferSize_, offsetPartial_, v17OrLess_);
          check |= std::abs(criterion[0] * dy + criterion[1] * dx + criterion[2]) < criterion[3];
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCGuardRing::excludePartial: Point " << point << " zside " << zside << " layer "
                               << layer << " wafer " << waferU << ":" << waferV << " partial type " << partial
                               << " type " << type << " check " << check;
#endif
  }
  return check;
}

bool HGCGuardRing::insidePolygon(double x, double y, const std::vector<std::pair<double, double> >& xyv) {
  int counter(0);
  double x1(xyv[0].first), y1(xyv[0].second);
  for (unsigned i1 = 1; i1 <= xyv.size(); i1++) {
    unsigned i2 = (i1 % xyv.size());
    double x2(xyv[i2].first), y2(xyv[i2].second);
    if (y > std::min(y1, y2)) {
      if (y <= std::max(y1, y2)) {
        if (x <= std::max(x1, x2)) {
          if (y1 != y2) {
            double xinter = (y - y1) * (x2 - x1) / (y2 - y1) + x1;
            if ((x1 == x2) || (x <= xinter))
              ++counter;
          }
        }
      }
    }
    x1 = x2;
    y1 = y2;
  }

  if (counter % 2 == 0)
    return false;
  else
    return true;
}
