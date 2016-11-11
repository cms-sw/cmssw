#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#define EDM_ML_DEBUG

FastTimeTopology::FastTimeTopology(const FastTimeDDDConstants& hdcons,
				   ForwardSubdetector sub,
				   int type) : hdcons_(hdcons), subdet_(sub),
					       type_(type) {
  nEtaZ_      = hdcons_.numberEtaZ(type_);
  nPhi_       = hdcons_.numberPhi(type_);
  kHGhalf_    = nEtaZ_*nPhi_;
  kHGeomHalf_ = 1;
  kSizeForDenseIndexing = (unsigned int)(2*kHGhalf_);
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeTopology initialized for subDetetcor " << subdet_
	    << " Type " << type_ << "  with " << nEtaZ_ 
	    << " cells along Z|Eta and " << nPhi_
	    << " cells along phi: total channels " << kSizeForDenseIndexing 
	    << ":" << (2*kHGeomHalf_) << std::endl;
#endif
}

uint32_t FastTimeTopology::detId2denseId(const DetId& id) const {

  FastTimeTopology::DecodedDetId id_ = decode(id);
  uint32_t idx = (uint32_t)(((id_.zside > 0) ? kHGhalf_ : 0) +
			    ((id_.iEtaZ-1)*nPhi_+id_.iPhi-1));
  return idx;
}

DetId FastTimeTopology::denseId2detId(uint32_t hi) const {

  if (validHashIndex(hi)) {
    FastTimeTopology::DecodedDetId id_;
    id_.iType  = type_;
    id_.zside  = ((int)(hi)<kHGhalf_ ? -1 : 1);
    int di     = ((int)(hi)%kHGhalf_);
    int iPhi   = (di%nPhi_);
    id_.iPhi   = iPhi+1;
    id_.iEtaZ  = (((di-iPhi)/nPhi_)+1);
    return encode(id_);
  } else {
    return DetId(0);
  }
}

uint32_t FastTimeTopology::detId2denseGeomId(const DetId& id) const {

  FastTimeTopology::DecodedDetId id_ = decode(id);
  uint32_t idx = (uint32_t)((id_.zside > 0) ? kHGeomHalf_ : 0);
  return idx;
}

bool FastTimeTopology::valid(const DetId& id) const {

  FastTimeTopology::DecodedDetId id_ = decode(id);
  bool flag = hdcons_.isValidXY(id_.iType,id_.iEtaZ,id_.iPhi);
  return flag;
}

DetId FastTimeTopology::offsetBy(const DetId startId, int nrStepsX,
				 int nrStepsY ) const {

  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)){
    DetId id = changeXY(startId,nrStepsX,nrStepsY);
    if (valid(id)) return id;
  }
  return DetId(0);
}

DetId FastTimeTopology::switchZSide(const DetId startId) const {

  if (startId.det() == DetId::Forward && startId.subdetId() == (int)(subdet_)){
    FastTimeTopology::DecodedDetId id_ = decode(startId);
    id_.zside  =-id_.zside;
    DetId id   = encode(id_);
    if (valid(id)) return id;
  }
  return DetId(0);
}

FastTimeTopology::DecodedDetId FastTimeTopology::geomDenseId2decId(const uint32_t& hi) const {

  FastTimeTopology::DecodedDetId id_;
  if (hi < totalGeomModules()) {
    id_.zside  = ((int)(hi)<kHGeomHalf_ ? -1 : 1);
  }
  return id_;
}

FastTimeTopology::DecodedDetId FastTimeTopology::decode(const DetId& startId) const {

  FastTimeTopology::DecodedDetId id_;
  FastTimeDetId id(startId);
  id_.iPhi   = id.iphi();
  id_.iEtaZ  = id.ieta();
  id_.iType  = id.type();
  id_.zside  = id.zside();
  id_.subdet = id.subdetId();
  return id_;
}

DetId FastTimeTopology::encode(const FastTimeTopology::DecodedDetId& id_) const {

  DetId id = FastTimeDetId(id_.iType,id_.iEtaZ,id_.iPhi,id_.zside);
  return id;
}

DetId FastTimeTopology::changeXY(const DetId& id, int nrStepsX,
				 int nrStepsY ) const {

  FastTimeTopology::DecodedDetId id_ = decode(id);
  int iEtaZ = id_.iEtaZ + nrStepsX;
  int iPhi  = id_.iPhi  + nrStepsY;
  if      (iPhi < 1)     iPhi += nPhi_;
  else if (iPhi > nPhi_) iPhi -= nPhi_;
  if (id_.iType == 1 && iEtaZ < 0) {
    iEtaZ = -iEtaZ;
    id_.zside = -id_.zside;
  }
  id_.iPhi   = iPhi;
  id_.iEtaZ  = iEtaZ;
  DetId nextPoint = encode(id_);
  if (valid(nextPoint)) return nextPoint;
  else                  return DetId(0);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(FastTimeTopology);
