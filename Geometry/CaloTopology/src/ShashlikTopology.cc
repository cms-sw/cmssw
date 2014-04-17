#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#define DebugLog

ShashlikTopology::ShashlikTopology(const ShashlikDDDConstants* sdcons) :
  sdcons_(sdcons) {
  smodules_ = sdcons->getSuperModules();
  modules_  = sdcons->getModules();
  nRows_    = (sdcons->getCols())/2;
  kEKhalf_  = smodules_*modules_*FIB_MAX*RO_MAX;
  kSizeForDenseIndexing = (unsigned int)(2*kEKhalf_);
#ifdef DebugLog
  std::cout << "ShashlikTopology initialized with " << smodules_
	    << " SuperModules " << modules_ << " modules and total channels "
	    << kSizeForDenseIndexing << std::endl;
#endif
}

uint32_t ShashlikTopology::detId2denseId(const DetId& id) const {
  EKDetId id_(id);
  std::pair<int,int> ismm = sdcons_->getSMM(id_.ix(),id_.iy());
  int iFib = id_.fiber();
  int iRO  = id_.readout();
  uint32_t idx = (uint32_t)((((id_.zside() > 0) ? kEKhalf_ : 0) +
			     ((((ismm.first-1)*modules_+ismm.second-1)*FIB_MAX+
			       iFib)*RO_MAX+iRO)));
  return idx;
}

DetId ShashlikTopology::denseId2detId(uint32_t hi) const {

  if (validHashIndex(hi)) {
    int iz ((int)(hi)<kEKhalf_ ? -1 : 1);
    int di ((int)(hi)%kEKhalf_);
    int ro (di%RO_MAX);
    int fib (((di-ro)/RO_MAX)%FIB_MAX);
    int iMD (((((di-ro)/RO_MAX)-fib)/FIB_MAX)%modules_+1);
    int iSM (((((di-ro)/RO_MAX)-fib)/FIB_MAX-iMD+1)/modules_+1);
    std::pair<int,int> ixy = sdcons_->getXY(iSM,iMD);
    return EKDetId(ixy.first, ixy.second, fib, ro, iz).rawId();
  } else {
    return DetId(0);
  }
}

bool ShashlikTopology::valid(const DetId& id) const {

  EKDetId id_(id);
  std::pair<int,int> ismm = sdcons_->getSMM(id_.ix(),id_.iy());
  int fib = id_.fiber();
  int ro  = id_.readout();
  bool flag = (ismm.first >= 1 && ismm.first <= smodules_ && ismm.second >= 1&&
	       ismm.second <= modules_ && fib >= 0 && fib < FIB_MAX &&
	       ro >= 0 && ro < RO_MAX);
  return flag;
}

bool ShashlikTopology::isNextToBoundary(EKDetId id) const {
  return isNextToDBoundary(id)  || isNextToRingBoundary(id) ;
}

bool ShashlikTopology::isNextToDBoundary(EKDetId id) const {
  // hardcoded values for D boundary
  return id.ix() == nRows_ || id.ix() == nRows_+1 ;
}

bool ShashlikTopology::isNextToRingBoundary(EKDetId id) const {
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      int ix = id.ix() + i;
      int iy = id.iy() + j;
      if (!valid(EKDetId(ix,iy,id.fiber(),id.readout(),id.zside()))) {
        return true;
      }
    }
  }
  return false;
}

DetId ShashlikTopology::offsetBy(const DetId startId, int nrStepsX,
				 int nrStepsY ) const {

  if (startId.det() == DetId::Ecal && startId.subdetId() == EcalShashlik) {
    EKDetId ekStartId(startId);
    int ix = ekStartId.ix() + nrStepsX;
    int iy = ekStartId.iy() + nrStepsY;
    EKDetId id = EKDetId(ix,iy,ekStartId.fiber(),ekStartId.readout(),ekStartId.zside());
    if (valid(id)) return id.rawId();
  }
  return DetId(0);
}

DetId ShashlikTopology::switchZSide(const DetId startId) const {

  if (startId.det() == DetId::Ecal && startId.subdetId() == EcalShashlik) {
    EKDetId ekStartId(startId);
    EKDetId id = EKDetId(ekStartId.ix(),ekStartId.iy(),ekStartId.fiber(),ekStartId.readout(),-ekStartId.zside());
    if (valid(id)) return id.rawId();
  }
  return DetId(0);
}

int ShashlikTopology::distanceX(const EKDetId& a,const EKDetId& b) {
  return abs(a.ix()-b.ix());
}

int ShashlikTopology::distanceY(const EKDetId& a,const EKDetId& b) {
  return abs(a.iy() - b.iy()); 
}

EKDetId ShashlikTopology::incrementIy(const EKDetId& id) const {
  EKDetId nextPoint = EKDetId(id.ix(),id.iy()+1,id.fiber(),id.readout(),id.zside());
  if (valid(nextPoint))
    return nextPoint;
  else
    return EKDetId(0);
} 

EKDetId ShashlikTopology::decrementIy(const EKDetId& id) const {
  EKDetId nextPoint = EKDetId(id.ix(),id.iy()-1,id.fiber(),id.readout(),id.zside());
  if (valid(nextPoint))
    return nextPoint;
  else
    return EKDetId(0);
} 

EKDetId ShashlikTopology::incrementIx(const EKDetId& id) const {
  EKDetId nextPoint = EKDetId(id.ix()+1,id.iy(),id.fiber(),id.readout(),id.zside());
  if (valid(nextPoint))
    return nextPoint;
  else
    return EKDetId(0);
} 

EKDetId ShashlikTopology::decrementIx(const EKDetId& id) const {
  EKDetId nextPoint = EKDetId(id.ix()-1,id.iy(),id.fiber(),id.readout(),id.zside());
  if (valid(nextPoint))
    return nextPoint;
  else
    return EKDetId(0);
} 

