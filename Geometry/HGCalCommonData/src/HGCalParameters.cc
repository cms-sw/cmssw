#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
//#define EDM_ML_DEBUG

HGCalParameters::HGCalParameters(const std::string& nam): name_(nam) { }

HGCalParameters::~HGCalParameters() { }

void HGCalParameters::fillModule(const HGCalParameters::hgtrap& mytr,
				 bool reco) {

  if (reco) {
    moduleLayR_.push_back(mytr.lay);
    moduleBlR_.push_back(mytr.bl);
    moduleTlR_.push_back(mytr.tl);
    moduleHR_.push_back(mytr.h);
    moduleDzR_.push_back(mytr.dz);
    moduleAlphaR_.push_back(mytr.alpha);
    moduleCellR_.push_back(mytr.cellSize);
  } else {
    moduleLayS_.push_back(mytr.lay);
    moduleBlS_.push_back(mytr.bl);
    moduleTlS_.push_back(mytr.tl);
    moduleHS_.push_back(mytr.h);
    moduleDzS_.push_back(mytr.dz);
    moduleAlphaS_.push_back(mytr.alpha);
    moduleCellS_.push_back(mytr.cellSize);
  }
}

HGCalParameters::hgtrap HGCalParameters::getModule(unsigned int k, 
						   bool reco) const {
  HGCalParameters::hgtrap mytr;
  if (reco) {
    if (k < moduleLayR_.size()) {
      mytr.lay      = moduleLayR_[k];
      mytr.bl       = moduleBlR_[k];
      mytr.tl       = moduleTlR_[k];
      mytr.h        = moduleHR_[k];
      mytr.dz       = moduleDzR_[k];
      mytr.alpha    = moduleAlphaR_[k];
      mytr.cellSize = moduleCellR_[k];
    } else {
      mytr.lay      = -1;
      mytr.bl  =  mytr.tl = mytr.h = mytr.dz = mytr.alpha = mytr.cellSize = 0;
    }
  } else {
    if (k < moduleLayS_.size()) {
      mytr.lay      = moduleLayS_[k];
      mytr.bl       = moduleBlS_[k];
      mytr.tl       = moduleTlS_[k];
      mytr.h        = moduleHS_[k];
      mytr.dz       = moduleDzS_[k];
      mytr.alpha    = moduleAlphaS_[k];
      mytr.cellSize = moduleCellS_[k];
    } else {
      mytr.lay      = -1;
      mytr.bl  =  mytr.tl = mytr.h = mytr.dz = mytr.alpha = mytr.cellSize = 0;
    }
  }
  return mytr;
}

void HGCalParameters::fillTrForm(const HGCalParameters::hgtrform& mytr) {

  int       zp    = (mytr.zp == 1) ? 1 : 0;
  uint32_t indx  = ((zp & kMaskZside) << kShiftZside);
  indx          |= ((mytr.lay & kMaskLayer) << kShiftLayer);
  indx          |= ((mytr.sec & kMaskSector) << kShiftSector);
  indx          |= ((mytr.subsec & kMaskSubSec) << kShiftSubSec);
#ifdef EDM_ML_DEBUG
  std::cout << "ZP " << zp << ":" << kMaskZside << ":" << kShiftZside 
	    << ((zp & kMaskZside) << kShiftZside) << " Lay " << mytr.lay 
	    << ":" << kMaskLayer << ":" << kShiftLayer << ":" 
	    << ((mytr.lay & kMaskLayer) << kShiftLayer) << " Sector " 
	    << mytr.sec << ":" << kMaskSector << ":" << kShiftSector << ":" 
	    << ((mytr.sec & kMaskSector) << kShiftSector) << " SubSec " 
	    << mytr.subsec << ":" << kMaskSubSec << ":" << kShiftSubSec << ":"
	    << ((mytr.subsec & kMaskSubSec) << kShiftSubSec) << " Index " 
	    << std::hex << indx << std::dec << std::endl;
#endif
  trformIndex_.push_back(indx);
  trformTranX_.push_back(mytr.h3v.x());
  trformTranY_.push_back(mytr.h3v.y());
  trformTranZ_.push_back(mytr.h3v.z());
  trformRotXX_.push_back(mytr.hr.xx());
  trformRotYX_.push_back(mytr.hr.yx());
  trformRotZX_.push_back(mytr.hr.zx());
  trformRotXY_.push_back(mytr.hr.xy());
  trformRotYY_.push_back(mytr.hr.yy());
  trformRotZY_.push_back(mytr.hr.zy());
  trformRotXZ_.push_back(mytr.hr.xz());
  trformRotYZ_.push_back(mytr.hr.yz());
  trformRotZZ_.push_back(mytr.hr.zz());
#ifdef EDM_ML_DEBUG
  unsigned int k = trformIndex_.size() - 1;
  std::cout << "HGCalParameters[" << k << "] Index " << std::hex
	    << trformIndex_[k] << std::dec << " (" << mytr.zp << ", "<< mytr.lay
	    << ", " << mytr.sec << ", " << mytr.subsec << ") Translation ("
	    << trformTranX_[k] << ", " << trformTranY_[k] << ", "
	    << trformTranZ_[k] << ") Rotation (" << trformRotXX_[k] << ", "
	    << trformRotYX_[k] << ", " << trformRotZX_[k] << ", "
	    << trformRotXY_[k] << ", " << trformRotYY_[k] << ", "
	    << trformRotZY_[k] << ", " << trformRotXZ_[k] << ", "
	    << trformRotYZ_[k] << ", " << trformRotZZ_[k] << std::endl;
#endif
}

HGCalParameters::hgtrform HGCalParameters::getTrForm(unsigned int k) const {

  HGCalParameters::hgtrform mytr;
  if (k < trformIndex_.size()) {
    int zp = ((trformIndex_[k] >> kShiftZside) & kMaskZside);
    mytr.zp    = (zp == 1) ? 1 : -1;
    mytr.lay   = ((trformIndex_[k] >> kShiftLayer) & kMaskLayer);
    mytr.sec   = ((trformIndex_[k] >> kShiftSector) & kMaskSector);
    mytr.subsec= ((trformIndex_[k] >> kShiftSubSec) & kMaskSubSec);
    mytr.h3v   = CLHEP::Hep3Vector(trformTranX_[k],trformTranY_[k],trformTranZ_[k]);
    const CLHEP::HepRep3x3 rotation(trformRotXX_[k],trformRotXY_[k],trformRotXZ_[k],
				    trformRotYX_[k],trformRotYY_[k],trformRotYZ_[k],
				    trformRotZX_[k],trformRotZY_[k],trformRotZZ_[k]);
    mytr.hr = CLHEP::HepRotation(rotation);
  } else {
    mytr.zp = mytr.lay = mytr.sec = mytr.subsec = 0;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalParameters[" << k << "] Index " << std::hex
	    << trformIndex_[k] << std::dec << " (" << mytr.zp << ", "<< mytr.lay
	    << ", " << mytr.sec << ", " << mytr.subsec << ") Translation ("
	    << mytr.h3v.x() << ", " << mytr.h3v.y() << ", " << mytr.h3v.z() 
	    << ") Rotation (" << mytr.hr.xx() << ", " << mytr.hr.yx() << ", " 
	    << mytr.hr.zx() << ", "  << mytr.hr.xy() << ", " << mytr.hr.yy() 
	    << ", " << mytr.hr.zy() << ", " << mytr.hr.xz() << ", "
	    << mytr.hr.yz() << ", " << mytr.hr.zz() << std::endl;
#endif
  return mytr;
}

void HGCalParameters::addTrForm(const CLHEP::Hep3Vector& h3v) {

  unsigned int k = trformTranX_.size();
  if (k > 0) {
    trformTranX_[k-1] += h3v.x();
    trformTranY_[k-1] += h3v.y();
    trformTranZ_[k-1] += h3v.z();
  }
}

void HGCalParameters::scaleTrForm(double scale) {
  unsigned int k = trformTranX_.size();
  if (k > 0) {
    trformTranX_[k-1] *= scale;
    trformTranY_[k-1] *= scale;
    trformTranZ_[k-1] *= scale;
  }
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalParameters);
