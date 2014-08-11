#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//#define DebugLog

HGCalTopology::HGCalTopology(const HGCalDDDConstants& hdcons, 
			     ForwardSubdetector subdet,
			     bool half) : hdcons_(hdcons), subdet_(subdet),
					  half_(half) {
  sectors_  = hdcons_.sectors();
  layers_   = hdcons_.layers(true);
  cells_    = hdcons_.maxCells(true);
  kHGhalf_  = sectors_*layers_*subSectors_*cells_ ;
  kHGeomHalf_ = (half_ ? (sectors_*layers_*subSectors_) : (sectors_*layers_));
  kSizeForDenseIndexing = (unsigned int)(2*kHGhalf_);
#ifdef DebugLog
  std::cout << "HGCalTopology initialized for subdetector " << subdet_ 
	    << " having half-chamber flag " << half_ << " with " << sectors_
	    << " Sectors " << layers_ << " Layers " << cells_
	    << " cells and total channels " << kSizeForDenseIndexing << ":"
	    << (2*kHGeomHalf_) << std::endl;
#endif
}

uint32_t HGCalTopology::detId2denseId(const DetId& id) const {

  HGCalTopology::DecodedDetId id_ = decode(id);
  int isubsec= (id_.iSubSec > 0) ? 1 : 0;
  uint32_t idx = (uint32_t)((((id_.zside > 0) ? kHGhalf_ : 0) +
			     ((((id_.iCell-1)*layers_+id_.iLay-1)*sectors_+
			       id_.iSec-1)*subSectors_+isubsec)));
  return idx;
}

DetId HGCalTopology::denseId2detId(uint32_t hi) const {

  if (validHashIndex(hi)) {
    HGCalTopology::DecodedDetId id_;
    id_.zside  = ((int)(hi)<kHGhalf_ ? -1 : 1);
    int di     = ((int)(hi)%kHGhalf_);
    int iSubSec= (di%subSectors_);
    id_.iSubSec= (iSubSec == 0 ? -1 : 1);
    id_.iSec   = (((di-iSubSec)/subSectors_)%sectors_+1);
    id_.iLay   = (((((di-iSubSec)/subSectors_)-id_.iSec+1)/sectors_)%layers_+1);
    id_.iCell  = (((((di-iSubSec)/subSectors_)-id_.iSec+1)/sectors_-id_.iLay+1)/layers_+1);
    return encode(id_);
  } else {
    return DetId(0);
  }
}

uint32_t HGCalTopology::detId2denseGeomId(const DetId& id) const {

  HGCalTopology::DecodedDetId id_ = decode(id);
  int isubsec= (half_ && id_.iSubSec > 0) ? 1 : 0;
  uint32_t idx = (uint32_t)(((id_.zside > 0) ? kHGeomHalf_ : 0) +
			    ((isubsec*layers_+id_.iLay-1)*sectors_+
			     id_.iSec-1));
  return idx;
}

bool HGCalTopology::valid(const DetId& id) const {

  HGCalTopology::DecodedDetId id_ = decode(id);
  bool flag = (id.det() == DetId::Forward && id.subdetId() == (int)(subdet_) &&
	       id_.iCell >= 0 && id_.iCell < cells_ && id_.iLay > 0 && 
	       id_.iLay <= layers_ && id_.iSec > 0 && id_.iSec <= sectors_);
  return flag;
}

DetId HGCalTopology::offsetBy(const DetId startId, int nrStepsX,
			      int nrStepsY ) const {

  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)){
    DetId id = changeXY(startId,nrStepsX,nrStepsY);
    if (valid(id)) return id.rawId();
  }
  return DetId(0);
}

DetId HGCalTopology::switchZSide(const DetId startId) const {

  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)){
    HGCalTopology::DecodedDetId id_ = decode(startId);
    id_.zside  =-id_.zside;
    DetId id   = encode(id_);
    if (valid(id)) return id.rawId();
  }
  return DetId(0);
}

HGCalTopology::DecodedDetId HGCalTopology::geomDenseId2decId(const uint32_t& hi) const {

  HGCalTopology::DecodedDetId id_;
  if (hi < totalGeomModules()) {
    id_.zside  = ((int)(hi)<kHGeomHalf_ ? -1 : 1);
    int di     = ((int)(hi)%kHGeomHalf_);
    int iSubSec= (di%subSectors_);
    id_.iSubSec= (iSubSec == 0 ? -1 : 1);
    id_.iSec   = (((di-iSubSec)/subSectors_)%sectors_+1);
    id_.iLay   = (((((di-iSubSec)/subSectors_)-id_.iSec+1)/sectors_)%layers_+1);
  }
  return id_;
}

HGCalTopology::DecodedDetId HGCalTopology::decode(const DetId& startId) const {

  HGCalTopology::DecodedDetId id_;
  if (subdet_ == HGCEE) {
    HGCEEDetId id(startId);
    id_.iCell  = id.cell();
    id_.iLay   = id.layer();
    id_.iSec   = id.sector();
    id_.iSubSec= id.subsector();
    id_.zside  = id.zside();
    id_.subdet = id.subdetId();
  } else {
    HGCHEDetId id(startId);
    id_.iCell  = id.cell();
    id_.iLay   = id.layer();
    id_.iSec   = id.sector();
    id_.iSubSec= id.subsector();
    id_.zside  = id.zside();
    id_.subdet = id.subdetId();
  }
  return id_;
}

DetId HGCalTopology::encode(const HGCalTopology::DecodedDetId& id_) const {

  int isubsec= (id_.iSubSec > 0) ? 1 : 0;
  DetId id;
  if (subdet_ == HGCEE) {
    id = HGCEEDetId(subdet_,id_.zside,id_.iLay,id_.iSec,isubsec,id_.iCell).rawId();
  } else {
    id = HGCHEDetId(subdet_,id_.zside,id_.iLay,id_.iSec,isubsec,id_.iCell).rawId();
  }
  return id;
}

DetId HGCalTopology::changeXY(const DetId& id, int nrStepsX,
			      int nrStepsY ) const {

  HGCalTopology::DecodedDetId id_ = decode(id);
  std::pair<int,int> kcell= hdcons_.newCell(id_.iCell,id_.iLay,id_.iSec,
					    id_.iSubSec,nrStepsX,nrStepsY,
					    half_);
  id_.iSubSec= kcell.second;
  id_.iSec   = (kcell.second > 0) ? kcell.second : -kcell.second;
  id_.iCell  = kcell.first;
  DetId nextPoint = encode(id_);
  if (valid(nextPoint)) return nextPoint;
  else                  return DetId(0);
}


DetId HGCalTopology::changeZ(const DetId& id, int nrStepsZ) const {

  HGCalTopology::DecodedDetId id_  = decode(id);
  std::pair<int,int> kcell = hdcons_.newCell(id_.iCell,id_.iLay,
					     id_.iSubSec,nrStepsZ,half_);
  id_.iLay    = kcell.second;
  id_.iCell   = kcell.first;
  DetId nextPoint = encode(id_);
  if (valid(nextPoint)) return nextPoint;
  else                  return DetId(0);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTopology);
