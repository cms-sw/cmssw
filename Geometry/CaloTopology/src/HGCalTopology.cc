#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//#define EDM_ML_DEBUG

HGCalTopology::HGCalTopology(const HGCalDDDConstants& hdcons, int det) : hdcons_(hdcons) {
  sectors_ = hdcons_.sectors();
  layers_ = hdcons_.layers(true);
  cells_ = hdcons_.maxCells(true);
  mode_ = hdcons_.geomMode();
  cellMax_ = hdcons_.maxCellUV();
  waferOff_ = hdcons_.waferUVMax();
  waferMax_ = 2 * waferOff_ + 1;
  kHGhalf_ = sectors_ * layers_ * cells_;
  firstLay_ = hdcons_.firstLayer();
  if (waferHexagon6()) {
    det_ = DetId::Forward;
    subdet_ = (ForwardSubdetector)(det);
    kHGeomHalf_ = sectors_ * layers_;
    types_ = 2;
  } else if (det == (int)(DetId::Forward)) {
    det_ = DetId::Forward;
    subdet_ = HFNose;
    kHGeomHalf_ = sectors_ * layers_;
    types_ = 3;
  } else if (tileTrapezoid()) {
    det_ = (DetId::Detector)(det);
    subdet_ = ForwardEmpty;
    kHGeomHalf_ = sectors_ * layers_ * cellMax_;
    types_ = 2;
  } else {
    det_ = (DetId::Detector)(det);
    subdet_ = ForwardEmpty;
    kHGeomHalf_ = sectors_ * layers_;
    types_ = 3;
  }
  kSizeForDenseIndexing = (unsigned int)(2 * kHGhalf_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTopology initialized for detector " << det << ":" << det_ << ":" << subdet_
                                << " having " << sectors_ << " Sectors, " << layers_ << " Layers from " << firstLay_
                                << ", " << cells_ << " cells and total channels " << kSizeForDenseIndexing << ":"
                                << (2 * kHGeomHalf_);
#endif
}

unsigned int HGCalTopology::allGeomModules() const {
  return (tileTrapezoid() ? (unsigned int)(2 * hdcons_.numberCells(true)) : (unsigned int)(2 * hdcons_.wafers()));
}

std::vector<DetId> HGCalTopology::neighbors(const DetId& idin) const {
  std::vector<DetId> ids;
  HGCalTopology::DecodedDetId id = decode(idin);
  if (waferHexagon8()) {
    HGCalTypes::CellType celltype = hdcons_.cellType(id.iType, id.iCell1, id.iCell2);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Type:WaferU:WaferV " << id.iType << ":" << id.iCell1 << ":" << id.iCell2
                                  << " CellType "
                                  << static_cast<std::underlying_type<HGCalTypes::CellType>::type>(celltype);
#endif
    switch (celltype) {
      case (HGCalTypes::CellType::CentralType): {
        // cell within the wafer
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 0";
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::BottomLeftEdge): {
        // bottom left edge
        int wu1(id.iSec1), wv1(id.iSec2 - 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 1 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 2 * N1 - 1, v1 + N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 2 * N1 - 1, v1 + N1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::LeftEdge): {
        // left edege
        int wu1(id.iSec1 + 1), wv1(id.iSec2);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int u1 = hdcons_.modifyUV(id.iCell1, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 2 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << u1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1, 2 * N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1 - 1, 2 * N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::TopLeftEdge): {
        // top left edge
        int wu1(id.iSec1 + 1), wv1(id.iSec2 + 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 3 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1 + 1, v1 + N1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1, v1 + N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::TopRightEdge): {
        // top right edge
        int wu1(id.iSec1), wv1(id.iSec2 + 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 4 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 0, v1 - N1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 0, v1 - N1 + 1);
        break;
      }
      case (HGCalTypes::CellType::RightEdge): {
        // right edge
        int wu1(id.iSec1 - 1), wv1(id.iSec2);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int u1 = hdcons_.modifyUV(id.iCell1, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 5 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << u1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 - N1, 0);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 - N1 + 1, 0);
        break;
      }
      case (HGCalTypes::CellType::BottomRightEdge): {
        // bottom right edge
        int wu1(id.iSec1 - 1), wv1(id.iSec2 - 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int u1 = hdcons_.modifyUV(id.iCell1, id.iType, t1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 6 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << u1;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1 - 1, u1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1, u1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::BottomCorner): {
        // bottom corner
        int wu1(id.iSec1), wv1(id.iSec2 - 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
        int wu2(id.iSec1 - 1), wv2(id.iSec2 - 1);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int u2 = hdcons_.modifyUV(id.iCell1, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 11 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1 << ":" << t2
                                      << ":" << N2 << ":" << u2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 2 * N1 - 1, v1 + N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 2 * N1 - 1, v1 + N1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 + N2, u2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::BottomLeftCorner): {
        // bottom left corner
        int wu1(id.iSec1 + 1), wv1(id.iSec2);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int u1 = hdcons_.modifyUV(id.iCell1, id.iType, t1);
        int wu2(id.iSec1), wv2(id.iSec2 - 1);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int v2 = hdcons_.modifyUV(id.iCell2, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 12 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << u1 << ":" << t2
                                      << ":" << N2 << ":" << v2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1, 2 * N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, 2 * N2 - 1, v2 + N2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, 2 * N2 - 1, v2 + N2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::TopLeftCorner): {
        // top left corner
        int wu1(id.iSec1 + 1), wv1(id.iSec2 + 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
        int wu2(id.iSec1 + 1), wv2(id.iSec2);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int u2 = hdcons_.modifyUV(id.iCell1, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 13 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1 << ":" << t2
                                      << ":" << N2 << ":" << u2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1 + 1, N1 + v1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1, N1 + v1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 + N2 - 1, 2 * N2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2 + 1);
        break;
      }
      case (HGCalTypes::CellType::TopCorner): {
        // top corner
        int wu1(id.iSec1 + 1), wv1(id.iSec2 + 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
        int wu2(id.iSec1), wv2(id.iSec2 + 1);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int v2 = hdcons_.modifyUV(id.iCell2, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 14 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1 << ":" << t2
                                      << ":" << N2 << ":" << v2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1 + 1, v1 + N1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, v1, v1 + N1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 + 1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, 0, v2 - N2 + 1);
        break;
      }
      case (HGCalTypes::CellType::TopRightCorner): {
        // top right corner
        int wu1(id.iSec1), wv1(id.iSec2 + 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int v1 = hdcons_.modifyUV(id.iCell2, id.iType, t1);
        int wu2(id.iSec1 - 1), wv2(id.iSec2);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int u2 = hdcons_.modifyUV(id.iCell1, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 15 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << v1 << ":" << t2
                                      << ":" << N2 << ":" << u2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, 0, v1 - N1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 - N2, 0);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 - N2 + 1, 0);
        break;
      }
      case (HGCalTypes::CellType::BottomRightCorner): {
        // bottom right corner
        int wu1(id.iSec1 - 1), wv1(id.iSec2 - 1);
        int t1 = hdcons_.getTypeHex(id.iLay, wu1, wv1);
        int N1 = hdcons_.getUVMax(t1);
        int u1 = hdcons_.modifyUV(id.iCell1, id.iType, t1);
        int wu2(id.iSec1 - 1), wv2(id.iSec2);
        int t2 = hdcons_.getTypeHex(id.iLay, wu2, wv2);
        int N2 = hdcons_.getUVMax(t2);
        int u2 = hdcons_.modifyUV(id.iCell1, id.iType, t2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Cell Type 16 "
                                      << ":" << wu1 << ":" << wv1 << ":" << t1 << ":" << N1 << ":" << u1 << ":" << t2
                                      << ":" << N2 << ":" << u2;
#endif
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 + 1, id.iCell2);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, id.iType, id.iLay, id.iSec1, id.iSec2, id.iCell1 - 1, id.iCell2 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t1, id.iLay, wu1, wv1, u1 + N1 - 1, u1 - 1);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 - N2, 0);
        addHGCSiliconId(ids, id.det, id.zSide, t2, id.iLay, wu2, wv2, u2 - N2 + 1, 0);
        break;
      }
      default:
        // Not valid u, v
        int N = hdcons_.getUVMax(id.iType);
        edm::LogWarning("HGCalGeom") << "u:v " << id.iCell1 << ":" << id.iCell2 << " Tests " << (id.iCell1 > 2 * N - 1)
                                     << ":" << (id.iCell2 > 2 * N - 1) << ":" << (id.iCell2 >= (id.iCell1 + N)) << ":"
                                     << (id.iCell1 > (id.iCell2 + N)) << " ERROR";
    }
  } else if (tileTrapezoid()) {
    int iphi1 = (id.iCell1 > 1) ? id.iCell1 - 1 : hdcons_.getUVMax(id.iType);
    int iphi2 = (id.iCell1 < hdcons_.getUVMax(id.iType)) ? id.iCell1 + 1 : 1;
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 - 1, id.iCell1);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 - 1, iphi1);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1, iphi1);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 + 1, iphi1);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 + 1, id.iCell1);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 + 1, iphi2);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1, iphi2);
    addHGCSCintillatorId(ids, id.zSide, id.iType, id.iLay, id.iSec1 - 1, iphi2);
  }
  return ids;
}

uint32_t HGCalTopology::detId2denseId(const DetId& idin) const {
  HGCalTopology::DecodedDetId id = decode(idin);
  uint32_t idx;
  if (waferHexagon6()) {
    int type = (id.iType > 0) ? 1 : 0;
    idx = (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
                     ((((id.iCell1 - 1) * layers_ + id.iLay - 1) * sectors_ + id.iSec1) * types_ + type));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Hex " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iCell1
                                  << ":" << id.iType << " Constants " << kHGeomHalf_ << ":" << layers_ << ":"
                                  << sectors_ << ":" << types_ << " o/p " << idx;
#endif
  } else if (tileTrapezoid()) {
    idx =
        (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
                   ((((id.iCell1 - 1) * layers_ + id.iLay - firstLay_) * sectors_ + id.iSec1 - 1) * types_ + id.iType));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Trap " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iCell1
                                  << ":" << id.iType << " Constants " << kHGeomHalf_ << ":" << layers_ << ":"
                                  << sectors_ << ":" << types_ << " o/p " << idx;
#endif
  } else {
    idx =
        (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
                   (((((id.iCell1 * cellMax_ + id.iCell2) * layers_ + id.iLay - 1) * waferMax_ + id.iSec1 + waferOff_) *
                         waferMax_ +
                     id.iSec2 + waferOff_) *
                        types_ +
                    id.iType));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Hex8 " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iSec2
                                  << ":" << id.iCell1 << ":" << id.iCell2 << ":" << id.iType << " Constants "
                                  << kHGeomHalf_ << ":" << cellMax_ << ":" << layers_ << ":" << waferMax_ << ":"
                                  << waferOff_ << ":" << types_ << " o/p " << idx;
#endif
  }
  return idx;
}

DetId HGCalTopology::denseId2detId(uint32_t hi) const {
  HGCalTopology::DecodedDetId id;
  if (validHashIndex(hi)) {
    id.zSide = ((int)(hi) < kHGhalf_ ? -1 : 1);
    int di = ((int)(hi) % kHGhalf_);
    if (waferHexagon6()) {
      int type = (di % types_);
      id.iType = (type == 0 ? -1 : 1);
      id.iSec1 = (((di - type) / types_) % sectors_);
      id.iLay = (((((di - type) / types_) - id.iSec1 + 1) / sectors_) % layers_ + 1);
      id.iCell1 = (((((di - type) / types_) - id.iSec1 + 1) / sectors_ - id.iLay + 1) / layers_ + 1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Hex " << hi << " o/p " << id.zSide << ":" << id.iLay << ":" << id.iType
                                    << ":" << id.iSec1 << ":" << id.iCell1;
#endif
    } else if (tileTrapezoid()) {
      int type = (di % types_);
      id.iType = type;
      id.iSec1 = (((di - type) / types_) % sectors_) + 1;
      id.iLay = (((((di - type) / types_) - id.iSec1 + 1) / sectors_) % layers_ + firstLay_);
      id.iCell1 = (((((di - type) / types_) - id.iSec1 + 1) / sectors_ - id.iLay + firstLay_) / layers_ + 1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Trap " << hi << " o/p " << id.zSide << ":" << id.iLay << ":" << id.iType
                                    << ":" << id.iSec1 << ":" << id.iCell1;
#endif
    } else {
      int type = (di % types_);
      id.iType = type;
      di = (di - type) / types_;
      id.iSec2 = (di % waferMax_) - waferOff_;
      di = (di - id.iSec2 - waferOff_) / waferMax_;
      id.iSec1 = (di % waferMax_) - waferOff_;
      di = (di - id.iSec1 - waferOff_) / waferMax_;
      id.iLay = (di % layers_) + 1;
      di = (di - id.iLay + 1) / layers_;
      id.iCell2 = (di % cellMax_);
      id.iCell1 = (di - id.iCell2) / cellMax_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Hex8 " << hi << " o/p " << id.zSide << ":" << id.iLay << ":" << id.iType
                                    << ":" << id.iSec1 << ":" << id.iSec2 << ":" << id.iCell1 << ":" << id.iCell2;
#endif
    }
  }
  return encode(id);
}

uint32_t HGCalTopology::detId2denseGeomId(const DetId& idin) const {
  HGCalTopology::DecodedDetId id = decode(idin);
  uint32_t idx;
  if (waferHexagon6()) {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) + (id.iLay - 1) * sectors_ + id.iSec1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iType
                                  << " Constants " << kHGeomHalf_ << ":" << layers_ << ":" << sectors_ << " o/p "
                                  << idx;
#endif
  } else if (tileTrapezoid()) {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) +
                     (((id.iLay - firstLay_) * sectors_ + id.iSec1 - 1) * cellMax_ + id.iCell1 - 1));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Trap I/P " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":"
                                  << id.iCell1 << ":" << id.iType << " Constants " << kHGeomHalf_ << ":" << layers_
                                  << ":" << firstLay_ << ":" << sectors_ << ":" << cellMax_ << " o/p " << idx;
#endif
  } else {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) +
                     (((id.iLay - 1) * waferMax_ + id.iSec1 + waferOff_) * waferMax_ + id.iSec2 + waferOff_));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Hex8 I/P " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":"
                                  << id.iSec2 << ":" << id.iType << " Constants " << kHGeomHalf_ << ":" << layers_
                                  << ":" << waferMax_ << ":" << waferOff_ << " o/p " << idx;
#endif
  }
  return idx;
}

bool HGCalTopology::valid(const DetId& idin) const {
  HGCalTopology::DecodedDetId id = decode(idin);
  bool flag;
  if (waferHexagon6()) {
    flag = (idin.det() == det_ && idin.subdetId() == (int)(subdet_) && id.iCell1 >= 0 && id.iCell1 < cells_ &&
            id.iLay > 0 && id.iLay <= layers_ && id.iSec1 >= 0 && id.iSec1 <= sectors_);
    if (flag)
      flag = hdcons_.isValidHex(id.iLay, id.iSec1, id.iCell1, true);
  } else if (tileTrapezoid()) {
    flag = ((idin.det() == det_) && hdcons_.isValidTrap(id.iLay, id.iSec1, id.iCell1));
  } else {
    flag = ((idin.det() == det_) && hdcons_.isValidHex8(id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2));
  }
  return flag;
}

bool HGCalTopology::valid(const DetId& idin, int cornerMin) const {
  if (waferHexagon8()) {
    HGCalTopology::DecodedDetId id = decode(idin);
    bool mask = (cornerMin < HGCalTypes::WaferCornerMin) ? false : hdcons_.maskCell(idin, cornerMin);
    bool flag = ((idin.det() == det_) &&
                 hdcons_.isValidHex8(
                     id.iLay, id.iSec1, id.iSec2, id.iCell1, id.iCell2, (cornerMin >= HGCalTypes::WaferCornerMin)));
    return (flag && (!mask));
  } else {
    return valid(idin);
  }
}

bool HGCalTopology::validModule(const DetId& idin, int cornerMin) const {
  if (idin.det() != det_) {
    return false;
  } else if ((idin.det() == DetId::HGCalEE) || (idin.det() == DetId::HGCalHSi)) {
    HGCalTopology::DecodedDetId id = decode(idin);
    return hdcons_.isValidHex8(id.iLay, id.iSec1, id.iSec2, (cornerMin >= HGCalTypes::WaferCornerMin));
  } else {
    return valid(idin);
  }
}

DetId HGCalTopology::offsetBy(const DetId startId, int nrStepsX, int nrStepsY) const {
  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)) {
    DetId id = changeXY(startId, nrStepsX, nrStepsY);
    if (valid(id))
      return id;
  }
  return DetId(0);
}

DetId HGCalTopology::switchZSide(const DetId startId) const {
  HGCalTopology::DecodedDetId id_ = decode(startId);
  id_.zSide = -id_.zSide;
  DetId id = encode(id_);
  if (valid(id))
    return id;
  else
    return DetId(0);
}

HGCalTopology::DecodedDetId HGCalTopology::geomDenseId2decId(const uint32_t& hi) const {
  HGCalTopology::DecodedDetId id;
  if (hi < totalGeomModules()) {
    id.zSide = ((int)(hi) < kHGeomHalf_ ? -1 : 1);
    int di = ((int)(hi) % kHGeomHalf_);
    if (waferHexagon6()) {
      id.iSec1 = (di % sectors_);
      di = (di - id.iSec1) / sectors_;
      id.iLay = (di % layers_) + 1;
      id.iType = ((di - id.iLay + 1) / layers_ == 0) ? -1 : 1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << hi << " O/P " << id.zSide << ":" << id.iType << ":" << id.iLay
                                    << ":" << id.iSec1;
#endif
    } else if (tileTrapezoid()) {
      id.iCell1 = (di % cellMax_) + 1;
      di = (di - id.iCell1 + 1) / cellMax_;
      id.iSec1 = (di % sectors_) + 1;
      di = (di - id.iSec1 + 1) / sectors_;
      id.iLay = (di % layers_) + firstLay_;
      id.iType = (di - id.iLay + firstLay_) / layers_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Trap I/P " << hi << " O/P " << id.zSide << ":" << id.iType << ":"
                                    << id.iLay << ":" << id.iSec1 << ":" << id.iCell1;
#endif
    } else {
      id.iSec2 = (di % waferMax_) - waferOff_;
      di = (di - id.iSec2 - waferOff_) / waferMax_;
      id.iSec1 = (di % waferMax_) - waferOff_;
      di = (di - id.iSec1 - waferOff_) / waferMax_;
      id.iLay = (di % layers_) + 1;
      id.iType = (di - id.iLay + 1) / layers_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Hex8 I/P " << hi << " O/P " << id.zSide << ":" << id.iType << ":"
                                    << id.iLay << ":" << id.iSec1 << ":" << id.iSec2;
#endif
    }
  }
  return id;
}

void HGCalTopology::addHGCSCintillatorId(
    std::vector<DetId>& ids, int zside, int type, int lay, int iradius, int iphi) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "addHGCSCintillatorId " << zside << ":" << type << ":" << lay << ":" << iradius
                                << ":" << iphi << " ==> Validity " << hdcons_.isValidTrap(lay, iradius, iphi);
#endif
  if (hdcons_.isValidTrap(lay, iradius, iphi)) {
    HGCScintillatorDetId id(type, lay, zside * iradius, iphi);
    ids.emplace_back(DetId(id));
  }
}

void HGCalTopology::addHGCSiliconId(
    std::vector<DetId>& ids, int det, int zside, int type, int lay, int waferU, int waferV, int cellU, int cellV) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "addHGCSiliconId " << det << ":" << zside << ":" << type << ":" << lay << ":"
                                << waferU << ":" << waferV << ":" << cellU << ":" << cellV << " ==> Validity "
                                << hdcons_.isValidHex8(lay, waferU, waferV, cellU, cellV);
#endif
  if (hdcons_.isValidHex8(lay, waferU, waferV, cellU, cellV)) {
    HGCSiliconDetId id((DetId::Detector)(det), zside, type, lay, waferU, waferV, cellU, cellV);
    ids.emplace_back(DetId(id));
  }
}

HGCalTopology::DecodedDetId HGCalTopology::decode(const DetId& startId) const {
  HGCalTopology::DecodedDetId idx;
  if (waferHexagon6()) {
    HGCalDetId id(startId);
    idx.iCell1 = id.cell();
    idx.iCell2 = 0;
    idx.iLay = id.layer();
    idx.iSec1 = id.wafer();
    idx.iSec2 = 0;
    idx.iType = id.waferType();
    idx.zSide = id.zside();
    idx.det = id.subdetId();
  } else if (tileTrapezoid()) {
    HGCScintillatorDetId id(startId);
    idx.iCell1 = id.iphi();
    idx.iCell2 = 0;
    idx.iLay = id.layer();
    idx.iSec1 = id.ietaAbs();
    idx.iSec2 = 0;
    idx.iType = id.type();
    idx.zSide = id.zside();
    idx.det = (int)(id.subdet());
  } else if (det_ == DetId::Forward && subdet_ == ForwardSubdetector::HFNose) {
    HFNoseDetId id(startId);
    idx.iCell1 = id.cellU();
    idx.iCell2 = id.cellV();
    idx.iLay = id.layer();
    idx.iSec1 = id.waferU();
    idx.iSec2 = id.waferV();
    idx.iType = id.type();
    idx.zSide = id.zside();
    idx.det = (int)(id.subdet());
  } else {
    HGCSiliconDetId id(startId);
    idx.iCell1 = id.cellU();
    idx.iCell2 = id.cellV();
    idx.iLay = id.layer();
    idx.iSec1 = id.waferU();
    idx.iSec2 = id.waferV();
    idx.iType = id.type();
    idx.zSide = id.zside();
    idx.det = (int)(id.subdet());
  }
  return idx;
}

DetId HGCalTopology::encode(const HGCalTopology::DecodedDetId& idx) const {
  DetId id;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Encode " << idx.det << ":" << idx.zSide << ":" << idx.iType << ":" << idx.iLay
                                << ":" << idx.iSec1 << ":" << idx.iSec2 << ":" << idx.iCell1 << ":" << idx.iCell2;
#endif
  if (waferHexagon6()) {
    id =
        HGCalDetId((ForwardSubdetector)(idx.det), idx.zSide, idx.iLay, ((idx.iType > 0) ? 1 : 0), idx.iSec1, idx.iCell1)
            .rawId();
  } else if (tileTrapezoid()) {
    id = HGCScintillatorDetId(idx.iType, idx.iLay, idx.zSide * idx.iSec1, idx.iCell1).rawId();
  } else if (det_ == DetId::Forward && subdet_ == ForwardSubdetector::HFNose) {
    id = HFNoseDetId(idx.zSide, idx.iType, idx.iLay, idx.iSec1, idx.iSec2, idx.iCell1, idx.iCell2).rawId();
  } else {
    id = HGCSiliconDetId(
             (DetId::Detector)(idx.det), idx.zSide, idx.iType, idx.iLay, idx.iSec1, idx.iSec2, idx.iCell1, idx.iCell2)
             .rawId();
  }
  return id;
}

DetId HGCalTopology::changeXY(const DetId& id, int nrStepsX, int nrStepsY) const { return DetId(); }

DetId HGCalTopology::changeZ(const DetId& id, int nrStepsZ) const { return DetId(); }

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTopology);
