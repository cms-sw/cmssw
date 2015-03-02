#include "Geometry/CaloTopology/interface/ShashlikTopology.h"

//#define DebugLog

void ShashlikTopology::init () {
  smodules_ = sdcons_.getSuperModules();
  modules_ = sdcons_.getModules();
  nRows_ = sdcons_.getCols()/2;
  kEKhalf_ = smodules_*modules_*FIB_MAX*RO_MAX;
  kSizeForDenseIndexing = (unsigned int)(2*kEKhalf_);
}

ShashlikTopology::ShashlikTopology(const ShashlikDDDConstants& sdcons)
  : sdcons_(sdcons)
{
  init ();
#ifdef DebugLog
  std::cout << "ShashlikTopology initialized with " << smodules_
	    << " SuperModules " << modules_ << " modules and total channels "
	    << kSizeForDenseIndexing << std::endl;
#endif
}

size_t ShashlikTopology::cell2denseId(const DetId& id) const {
  EKDetId id_(id);
  std::pair<int,int> ismm = sdcons_.getSMM(id_.ix(),id_.iy());
  size_t idx = (size_t) (((id_.zside()>0?1:0)*smodules_+
			  (ismm.first-1))*modules_+
			 (ismm.second-1));
  return idx;
}

DetId ShashlikTopology::denseId2cell(size_t hi) const {

  if (hi<cellHashSize()) {
    int iMD = hi%modules_+1;
    int iSM = (hi/=modules_)%smodules_+1;
    int iz = (hi/=smodules_)%2;
    if (iz==0) iz = -1;
    std::pair<int,int> ixy = sdcons_.getXY(iSM,iMD);
    return EKDetId(ixy.first, ixy.second, 0, 0, iz).rawId();
  } else {
    return DetId(0);
  }
}

unsigned int ShashlikTopology::detId2denseId(const DetId& id) const {
  EKDetId id_(id);
  int iFib = id_.fiber();
  int iRO  = id_.readout();
  uint32_t idx = cell2denseId (id);
  idx += (uint32_t) ((iRO*FIB_MAX+iFib)*modules_);
  return idx;
}

DetId ShashlikTopology::denseId2detId(unsigned int hi) const {
  
  if (hi<detIdHashSize()) {
    EKDetId cell = denseId2cell(hi%detIdHashSize());
    int fib = (hi/=cellHashSize())%FIB_MAX;
    int ro = (hi/=FIB_MAX)%RO_MAX;
    return EKDetId (cell.ix(), cell.iy(), fib, ro, cell.zside()).rawId();
  } else {
    return DetId(0);
  }
}

bool ShashlikTopology::valid(const DetId& id) const {

  EKDetId id_(id);
  int fib = id_.fiber();
  int ro  = id_.readout();
  bool flag = validXY (id_.ix(),id_.iy()) && 
    fib >= 0 && fib < FIB_MAX &&
    ro >= 0 && ro < RO_MAX;
  return flag;
}

bool ShashlikTopology::validXY(int ix, int iy) const {
  std::pair<int,int> ismm = sdcons_.getSMM(ix,iy, true);
  bool flag = (ismm.first >= 1 && ismm.first <= smodules_ && ismm.second >= 1&&
	       ismm.second <= modules_);
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

