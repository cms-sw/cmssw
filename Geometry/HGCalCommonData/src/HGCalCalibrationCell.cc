#include "Geometry/HGCalCommonData/interface/HGCalCalibrationCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <sstream>

//#define EDM_ML_DEBUG

HGCalCalibrationCell::HGCalCalibrationCell(const HGCalDDDConstants* cons) : cons_(cons) {
  wafer_ = std::make_unique<HGCalCell>(
      cons_->getParameter()->waferSize_, cons_->getParameter()->nCellsFine_, cons_->getParameter()->nCellsCoarse_);

  radius_[0] = cons_->getParameter()->calibCellRHD_;
  cells_[0].insert(
      cells_[0].end(), cons_->getParameter()->calibCellFullHD_.begin(), cons_->getParameter()->calibCellFullHD_.end());
  cells_[1].insert(
      cells_[1].end(), cons_->getParameter()->calibCellPartHD_.begin(), cons_->getParameter()->calibCellPartHD_.end());
  radius_[1] = cons_->getParameter()->calibCellRLD_;
  cells_[2].insert(
      cells_[2].end(), cons_->getParameter()->calibCellFullLD_.begin(), cons_->getParameter()->calibCellFullLD_.end());
  cells_[3].insert(
      cells_[3].end(), cons_->getParameter()->calibCellPartLD_.begin(), cons_->getParameter()->calibCellPartLD_.end());
  for (int k = 0; k < 2; ++k)
    radius_[k] *= radius_[k];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCalibrationCell: " << cells_[0].size() << " HD calibration cells of radius "
                                << std::sqrt(radius_[0]);
  for (unsigned int k = 0; k < cells_[0].size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << cells_[0][k] << ":" << cells_[1][k];
  edm::LogVerbatim("HGCalGeom") << "HGCalCalibrationCell: " << cells_[2].size() << " LD calibration cells of radius "
                                << std::sqrt(radius_[1]);
  for (unsigned int k = 0; k < cells_[2].size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << cells_[2][k] << ":" << cells_[3][k];
#endif
}

int HGCalCalibrationCell::findCell(
    int zside, int layer, int waferU, int waferV, int cellUV, const std::pair<double, double>& xy) const {
  const auto& info = cons_->waferInfo(layer, waferU, waferV);
  int ld = (info.type == HGCalTypes::WaferFineThin) ? 1 : 0;
  int part = (info.part == HGCalTypes::WaferFull) ? 1 : 0;
  int indx = 2 * ld + part;
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  st1 << "HGCalCalibrationCell::findCell::input " << layer << ":" << waferU << ":" << waferV << ":" << cellUV << ":"
      << xy.first << ":" << xy.second << " Type:Part " << info.type << ":" << info.part << ":" << ld << ":" << part
      << ":" << indx;
#endif
  bool ok = (std::find(cells_[indx].begin(), cells_[indx].end(), cellUV) != cells_[indx].end());
  int retval(-1);
  if (ok) {
    int layertype = (cons_->layerType(layer) == HGCalTypes::WaferCenterB) ? 1 : 0;
    int place = HGCalCell::cellPlacementIndex(zside, layertype, info.orient);
    int cellU = (cellUV / 100) % 100;
    int cellV = cellUV % 100;
    auto xyc = wafer_->cellUV2XY1(cellU, cellV, place, (1 - ld));
    double xx = xy.first - xyc.first;
    double yy = xy.second - xyc.second;
    double r2 = (xx * xx + yy * yy);
    retval = (r2 > radius_[ld]) ? 0 : 1;
#ifdef EDM_ML_DEBUG
    st1 << " layertype:place:cellU:cellV " << layertype << ":" << place << ":" << cellU << ":" << cellV << " xx "
        << xy.first << ":" << xyc.first << ":" << xx << " yy " << xy.second << ":" << xyc.second << ":" << yy << " R2 "
        << r2 << ":" << radius_[ld];
#endif
  }
#ifdef EDM_ML_DEBUG
  st1 << " Return Value " << ok << ":" << retval;
  edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
  return retval;
}
