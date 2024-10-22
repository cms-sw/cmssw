#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HGCalTBTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//#define EDM_ML_DEBUG

HGCalTBTopology::HGCalTBTopology(const HGCalTBDDDConstants* hdcons, int det) : hdcons_(hdcons) {
  sectors_ = hdcons_->sectors();
  layers_ = hdcons_->layers(true);
  cells_ = hdcons_->maxCells(true);
  mode_ = hdcons_->geomMode();
  kHGhalf_ = sectors_ * layers_ * cells_;
  firstLay_ = hdcons_->firstLayer();
  det_ = DetId::Forward;
  subdet_ = static_cast<ForwardSubdetector>(det);
  kHGeomHalf_ = sectors_ * layers_;
  types_ = 2;
  kHGhalfType_ = sectors_ * layers_ * cells_ * types_;
  kSizeForDenseIndexing = static_cast<unsigned int>(2 * kHGhalf_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBTopology initialized for detector " << det << ":" << det_ << ":" << subdet_
                                << " having " << sectors_ << " Sectors, " << layers_ << " Layers from " << firstLay_
                                << ", " << cells_ << " cells and total channels " << kSizeForDenseIndexing << ":"
                                << (2 * kHGeomHalf_);
#endif
}

HGCalTBTopology::~HGCalTBTopology() {}

std::vector<DetId> HGCalTBTopology::neighbors(DetId idin) const {
  std::vector<DetId> ids = north(idin);
  for (const auto& id : south(idin))
    ids.emplace_back(id);
  for (const auto& id : east(idin))
    ids.emplace_back(id);
  for (const auto& id : west(idin))
    ids.emplace_back(id);
  return ids;
}

unsigned int HGCalTBTopology::allGeomModules() const { return (static_cast<unsigned int>(2 * hdcons_->wafers())); }

DetId HGCalTBTopology::denseId2detId(uint32_t hi) const {
  HGCalTBTopology::DecodedDetId id;
  if (validHashIndex(hi)) {
    id.zSide = ((int)(hi) < kHGhalfType_ ? -1 : 1);
    int di = ((int)(hi) % kHGhalfType_);
    int type = (di % types_);
    id.iType = (type == 0 ? -1 : 1);
    id.iSec1 = (((di - type) / types_) % sectors_);
    id.iLay = (((((di - type) / types_) - id.iSec1 + 1) / sectors_) % layers_ + 1);
    id.iCell1 = (((((di - type) / types_) - id.iSec1 + 1) / sectors_ - id.iLay + 1) / layers_ + 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Hex " << hi << " o/p " << id.zSide << ":" << id.iLay << ":" << id.iType
                                  << ":" << id.iSec1 << ":" << id.iCell1;
#endif
  }
  return encode(id);
}

uint32_t HGCalTBTopology::detId2denseId(const DetId& idin) const {
  HGCalTBTopology::DecodedDetId id = decode(idin);
  int type = (id.iType > 0) ? 1 : 0;
  uint32_t idx =
      static_cast<uint32_t>(((id.zSide > 0) ? kHGhalfType_ : 0) +
                            ((((id.iCell1 - 1) * layers_ + id.iLay - 1) * sectors_ + id.iSec1) * types_ + type));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input Hex " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iCell1
                                << ":" << id.iType << " Constants " << kHGeomHalf_ << ":" << layers_ << ":" << sectors_
                                << ":" << types_ << " o/p " << idx;
#endif
  return idx;
}

uint32_t HGCalTBTopology::detId2denseGeomId(const DetId& idin) const {
  HGCalTBTopology::DecodedDetId id = decode(idin);
  uint32_t idx;
  idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) + (id.iLay - 1) * sectors_ + id.iSec1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << id.zSide << ":" << id.iLay << ":" << id.iSec1 << ":" << id.iType
                                << " Constants " << kHGeomHalf_ << ":" << layers_ << ":" << sectors_ << " o/p " << idx;
#endif
  return idx;
}

bool HGCalTBTopology::valid(const DetId& idin) const {
  HGCalTBTopology::DecodedDetId id = decode(idin);
  bool flag;
  flag = (idin.det() == det_ && idin.subdetId() == (int)(subdet_) && id.iCell1 >= 0 && id.iCell1 < cells_ &&
          id.iLay > 0 && id.iLay <= layers_ && id.iSec1 >= 0 && id.iSec1 <= sectors_);
  if (flag)
    flag = hdcons_->isValidHex(id.iLay, id.iSec1, id.iCell1, true);
  return flag;
}

DetId HGCalTBTopology::offsetBy(const DetId startId, int nrStepsX, int nrStepsY) const {
  if (startId.det() == DetId::Forward && startId.subdetId() == static_cast<int>(subdet_)) {
    DetId id = changeXY(startId, nrStepsX, nrStepsY);
    if (valid(id))
      return id;
  }
  return DetId(0);
}

DetId HGCalTBTopology::switchZSide(const DetId startId) const {
  HGCalTBTopology::DecodedDetId id_ = decode(startId);
  id_.zSide = -id_.zSide;
  DetId id = encode(id_);
  if (valid(id))
    return id;
  else
    return DetId(0);
}

HGCalTBTopology::DecodedDetId HGCalTBTopology::geomDenseId2decId(const uint32_t& hi) const {
  HGCalTBTopology::DecodedDetId id;
  if (hi < totalGeomModules()) {
    id.zSide = ((int)(hi) < kHGeomHalf_ ? -1 : 1);
    int di = ((int)(hi) % kHGeomHalf_);
    id.iSec1 = (di % sectors_);
    di = (di - id.iSec1) / sectors_;
    id.iLay = (di % layers_) + 1;
    id.iType = ((di - id.iLay + 1) / layers_ == 0) ? -1 : 1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << hi << " O/P " << id.zSide << ":" << id.iType << ":" << id.iLay
                                  << ":" << id.iSec1;
#endif
  }
  return id;
}

HGCalTBTopology::DecodedDetId HGCalTBTopology::decode(const DetId& startId) const {
  HGCalTBTopology::DecodedDetId idx;
  HGCalDetId id(startId);
  idx.iCell1 = id.cell();
  idx.iCell2 = 0;
  idx.iLay = id.layer();
  idx.iSec1 = id.wafer();
  idx.iSec2 = 0;
  idx.iType = id.waferType();
  idx.zSide = id.zside();
  idx.det = id.subdetId();
  return idx;
}

DetId HGCalTBTopology::encode(const HGCalTBTopology::DecodedDetId& idx) const {
  DetId id;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeomX") << "Encode " << idx.det << ":" << idx.zSide << ":" << idx.iType << ":" << idx.iLay
                                 << ":" << idx.iSec1 << ":" << idx.iSec2 << ":" << idx.iCell1 << ":" << idx.iCell2;
#endif
  id = HGCalDetId((ForwardSubdetector)(idx.det), idx.zSide, idx.iLay, ((idx.iType > 0) ? 1 : 0), idx.iSec1, idx.iCell1)
           .rawId();
  return id;
}

DetId HGCalTBTopology::changeXY(const DetId& id, int nrStepsX, int nrStepsY) const { return DetId(); }

DetId HGCalTBTopology::changeZ(const DetId& id, int nrStepsZ) const { return DetId(); }

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTBTopology);
