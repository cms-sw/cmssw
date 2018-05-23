#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//#define EDM_ML_DEBUG

HGCalTopology::HGCalTopology(const HGCalDDDConstants& hdcons, 
			     int det) : hdcons_(hdcons) {

  sectors_  = hdcons_.sectors();
  layers_   = hdcons_.layers(true);
  cells_    = hdcons_.maxCells(true);
  mode_     = hdcons_.geomMode();
  cellMax_  = hdcons_.maxCellUV();
  waferOff_ = hdcons_.waferUVMax();
  waferMax_ = 2*waferOff_ + 1;
  kHGhalf_  = sectors_*layers_*cells_ ;
  firstLay_ = hdcons_.firstLayer();
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    det_        = DetId::Forward;
    subdet_     = (ForwardSubdetector)(det);
    kHGeomHalf_ = sectors_*layers_;
    types_      = 2;
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    det_        = (DetId::Detector)(det);
    subdet_     = ForwardEmpty;
    kHGeomHalf_ = sectors_*layers_*cells_;
    types_      = 2;
  } else {
    det_        = (DetId::Detector)(det);
    subdet_     = ForwardEmpty;
    kHGeomHalf_ = sectors_*layers_;
    types_      = 3;
  }
  kSizeForDenseIndexing = (unsigned int)(2*kHGhalf_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTopology initialized for detector " 
				<< det_ << ":" << subdet_ << " having " 
				<< sectors_  << " Sectors, " << layers_ 
				<< " Layers from " << firstLay_ << ", " 
				<< cells_ << " cells and total channels " 
				<< kSizeForDenseIndexing << ":"
				<< (2*kHGeomHalf_) << std::endl;
#endif
}

unsigned int HGCalTopology::allGeomModules() const {
  return ((mode_ == HGCalGeometryMode::Trapezoid) ? 
	  (unsigned int)(2*hdcons_.numberCells(true)) : 
	  (unsigned int)(2*hdcons_.wafers()));
}

uint32_t HGCalTopology::detId2denseId(const DetId& idin) const {

  HGCalTopology::DecodedDetId id = decode(idin);
  uint32_t idx;
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    int type = (id.iType > 0) ? 1 : 0;
    idx = (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
		     ((((id.iCell1-1)*layers_+id.iLay-1)*sectors_+
		       id.iSec1)*types_+type));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Hex " << id.zSide << ":" << id.iLay 
				  << ":" << id.iSec1 << ":" << id.iCell1
				  << ":" << id.iType << " Constants " 
				  << kHGeomHalf_ << ":" << layers_ << ":" 
				  << sectors_ << ":" << types_<< " o/p " 
				  << idx;
#endif
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    idx = (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
		     ((((id.iCell1-1)*layers_+id.iLay-firstLay_)*sectors_+
		       id.iSec1-1)*types_+id.iType));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Trap " << id.zSide << ":" << id.iLay 
				  << ":" << id.iSec1 << ":" << id.iCell1
				  << ":" << id.iType << " Constants " 
				  << kHGeomHalf_ << ":" << layers_ << ":" 
				  << sectors_ << ":" << types_<< " o/p " 
				  << idx;
#endif
  } else {
    idx = (uint32_t)(((id.zSide > 0) ? kHGhalf_ : 0) +
		     (((((id.iCell1*cellMax_+id.iCell2)*layers_+
			 id.iLay-1)*waferMax_+id.iSec1+waferOff_)*
		       waferMax_+id.iSec2+waferOff_)*types_+id.iType));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Input Hex8 " << id.zSide << ":" << id.iLay 
				  << ":" << id.iSec1 << ":" << id.iSec2 << ":"
				  << id.iCell1 << ":" << id.iCell2 << ":"
				  << id.iType << " Constants " << kHGeomHalf_ 
				  << ":" << cellMax_ << ":" << layers_ << ":" 
				  << waferMax_ << ":" << waferOff_ << ":" 
				  << types_<< " o/p " << idx;
#endif
  }
  return idx;
}

DetId HGCalTopology::denseId2detId(uint32_t hi) const {

  HGCalTopology::DecodedDetId id;
  if (validHashIndex(hi)) {
    id.zSide  = ((int)(hi)<kHGhalf_ ? -1 : 1);
    int di     = ((int)(hi)%kHGhalf_);
    if ((mode_ == HGCalGeometryMode::Hexagon) || 
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      int type  = (di%types_);
      id.iType = (type == 0 ? -1 : 1);
      id.iSec1 = (((di-type)/types_)%sectors_);
      id.iLay  = (((((di-type)/types_)-id.iSec1+1)/sectors_)%layers_+1);
      id.iCell1= (((((di-type)/types_)-id.iSec1+1)/sectors_-id.iLay+1)/layers_+1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Hex " << hi << " o/p " << id.zSide 
				    << ":" << id.iLay << ":" << id.iType
				    << ":" << id.iSec1 << ":" << id.iCell1;
#endif
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      int type  = (di%types_);
      id.iType = type;
      id.iSec1 = (((di-type)/types_)%sectors_)+1;
      id.iLay  = (((((di-type)/types_)-id.iSec1+1)/sectors_)%layers_+firstLay_);
      id.iCell1= (((((di-type)/types_)-id.iSec1+1)/sectors_-id.iLay+firstLay_)/layers_+1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Trap " << hi << " o/p " << id.zSide 
				    << ":" << id.iLay << ":" << id.iType
				    << ":" << id.iSec1 << ":" << id.iCell1;
#endif
    } else {
      int type  = (di%types_);
      id.iType = type;
      di        = (di-type)/types_;
      id.iSec2 = (di%waferMax_)-waferOff_;
      di        = (di-id.iSec2-waferOff_)/waferMax_;
      id.iSec1 = (di%waferMax_)-waferOff_;
      di        = (di-id.iSec1-waferOff_)/waferMax_;
      id.iLay  = (di%layers_)+1;
      di        = (di-id.iLay+1)/layers_;
      id.iCell2 = (di%cellMax_);
      id.iCell1 = (di-id.iCell2)/cellMax_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Input Hex8 " << hi << " o/p " << id.zSide 
				    << ":" << id.iLay << ":" << id.iType
				    << ":" << id.iSec1 << ":" << id.iSec2
				    << ":" << id.iCell1 << ":" << id.iCell2;
#endif
    }
  }
  return encode(id);
}

uint32_t HGCalTopology::detId2denseGeomId(const DetId& idin) const {

  HGCalTopology::DecodedDetId id = decode(idin);
  uint32_t idx;
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) +
		     (id.iLay-1)*sectors_+id.iSec1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << id.zSide << ":" << id.iLay
				  << ":" << id.iSec1 << ":" << id.iType 
				  << " Constants " << kHGeomHalf_ << ":" 
				  << layers_ << ":" << sectors_ << " o/p " 
				  << idx;
#endif
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) +
		     (((id.iLay-firstLay_)*sectors_+id.iSec1-1)*cellMax_+
		      id.iCell1-1));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Trap I/P " << id.zSide << ":" 
				  << id.iLay  << ":" << id.iSec1 << ":" 
				  << id.iCell1 << ":" << id.iType << " Constants "
				  << kHGeomHalf_ << ":" << layers_ << ":" 
				  << firstLay_ << ":" << sectors_ << ":"
				  << cellMax_ << " o/p " << idx;
#endif
  } else {
    idx = (uint32_t)(((id.zSide > 0) ? kHGeomHalf_ : 0) +
		     (((id.iLay-1)*waferMax_+id.iSec1+waferOff_)*waferMax_+
		      id.iSec2+waferOff_));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Geom Hex8 I/P " << id.zSide << ":" 
				  << id.iLay  << ":" << id.iSec1 << ":" 
				  << id.iSec2 << ":" << id.iType << " Constants " 
				  << kHGeomHalf_ << ":" << layers_ << ":"
				  << waferMax_ << ":" << waferOff_ << " o/p "
				  << idx;
#endif
  }
  return idx;
}

bool HGCalTopology::valid(const DetId& idin) const {

  HGCalTopology::DecodedDetId id = decode(idin);
  bool flag;
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
	     (mode_ == HGCalGeometryMode::HexagonFull)) {
    flag = (idin.det() == det_ && idin.subdetId() == (int)(subdet_) &&
	    id.iCell1 >= 0 && id.iCell1 < cells_ && id.iLay > 0 && 
	    id.iLay <= layers_ && id.iSec1 >= 0 && id.iSec1 <= sectors_);
    if (flag) flag = hdcons_.isValidHex(id.iLay,id.iSec1,id.iCell1,true);
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    flag = ((idin.det() == det_) && 
	    hdcons_.isValidTrap(id.iLay,id.iSec1,id.iCell1));
  } else {
    flag = ((idin.det() == det_) && 
	    hdcons_.isValidHex8(id.iLay,id.iSec1,id.iSec2,id.iCell1,id.iCell2));
  }
  return flag;
}

DetId HGCalTopology::offsetBy(const DetId startId, int nrStepsX,
			      int nrStepsY ) const {

  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)){
    DetId id = changeXY(startId,nrStepsX,nrStepsY);
    if (valid(id)) return id;
  }
  return DetId(0);
}

DetId HGCalTopology::switchZSide(const DetId startId) const {

  HGCalTopology::DecodedDetId id_ = decode(startId);
  id_.zSide  =-id_.zSide;
  DetId id   = encode(id_);
  if (valid(id)) return id;
  else           return DetId(0);
}

HGCalTopology::DecodedDetId HGCalTopology::geomDenseId2decId(const uint32_t& hi) const {

  HGCalTopology::DecodedDetId id;
  if (hi < totalGeomModules()) {
    id.zSide  = ((int)(hi)<kHGeomHalf_ ? -1 : 1);
    int di    = ((int)(hi)%kHGeomHalf_);
    if ((mode_ == HGCalGeometryMode::Hexagon) || 
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      id.iSec1  = (di%sectors_);
      di        = (di-id.iSec1)/sectors_;
      id.iLay   = (di%layers_)+1;
      id.iType  = ((di-id.iLay+1)/layers_ == 0) ? -1 : 1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Hex I/P " << hi << " O/P " 
				    << id.zSide  << ":" << id.iType << ":" 
				    << id.iLay << ":" << id.iSec1;
#endif
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      id.iCell1 = (di%cellMax_)+1;
      di        = (di-id.iCell1+1)/cellMax_;
      id.iSec1  = (di%sectors_)+1;
      di        = (di-id.iSec1+1)/sectors_;
      id.iLay   = (di%layers_) + firstLay_;
      id.iType  = (di-id.iLay+firstLay_)/layers_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Trap I/P " << hi << " O/P " 
				    << id.zSide << ":" << id.iType << ":" 
				    << id.iLay << ":" << id.iSec1 << ":" 
				    << id.iCell1;
#endif
    } else {
      id.iSec2  = (di%waferMax_)-waferOff_;
      di        = (di-id.iSec2-waferOff_)/waferMax_;
      id.iSec1  = (di%waferMax_)-waferOff_;
      di        = (di-id.iSec1-waferOff_)/waferMax_;
      id.iLay   = (di%layers_)+1;
      id.iType  = (di-id.iLay+1)/layers_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Geom Hex8 I/P " << hi << " O/P " 
				    << id.zSide << ":" << id.iType << ":" 
				    << id.iLay << ":" << id.iSec1 << ":" 
				    << id.iSec2;
#endif
    }
  }
  return id;
}

HGCalTopology::DecodedDetId HGCalTopology::decode(const DetId& startId) const {

  HGCalTopology::DecodedDetId idx;
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    HGCalDetId id(startId);
    idx.iCell1 = id.cell();
    idx.iCell2 = 0;
    idx.iLay   = id.layer();
    idx.iSec1  = id.wafer();
    idx.iSec2  = 0;
    idx.iType  = id.waferType();
    idx.zSide  = id.zside();
    idx.det    = id.subdetId();
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    HGCScintillatorDetId id(startId);
    idx.iCell1 = id.iphi();
    idx.iCell2 = 0;
    idx.iLay   = id.layer();
    idx.iSec1  = id.ietaAbs();
    idx.iSec2  = 0;
    idx.iType  = id.type();
    idx.zSide  = id.zside();
    idx.det    = (int)(id.subdet());
  } else {
    HGCSiliconDetId id(startId);
    idx.iCell1 = id.cellU();
    idx.iCell2 = id.cellV();
    idx.iLay   = id.layer();
    idx.iSec1  = id.waferU();
    idx.iSec2  = id.waferV();
    idx.iType  = id.type();
    idx.zSide  = id.zside();
    idx.det    = (int)(id.subdet());
  }
  return idx;
}

DetId HGCalTopology::encode(const HGCalTopology::DecodedDetId& idx) const {

  DetId id;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    id = HGCalDetId((ForwardSubdetector)(idx.det),idx.zSide,idx.iLay,((idx.iType > 0) ? 1 : 0),idx.iSec1,idx.iCell1).rawId();
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    id = HGCScintillatorDetId(idx.iType,idx.iLay,idx.zSide*idx.iSec1,idx.iCell1).rawId();
  } else {
    id = HGCSiliconDetId((DetId::Detector)(idx.det),idx.zSide,idx.iType,idx.iLay,idx.iSec1,idx.iSec2,idx.iCell1,idx.iCell2).rawId();
  }
  return id;
}

DetId HGCalTopology::changeXY(const DetId& id, int nrStepsX,
			      int nrStepsY ) const {

  return DetId();
}


DetId HGCalTopology::changeZ(const DetId& id, int nrStepsZ) const {

  return DetId();
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTopology);
