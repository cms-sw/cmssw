#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <array>

//#define EDM_ML_DEBUG

HGCalTBParameters::HGCalTBParameters(const std::string& nam) : name_(nam), nCells_(0) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Construct HGCalTBParameters for " << name_;
#endif
}

void HGCalTBParameters::fillModule(const HGCalTBParameters::hgtrap& mytr, bool reco) {
  if (reco) {
    moduleLayR_.emplace_back(mytr.lay);
    moduleBlR_.emplace_back(mytr.bl);
    moduleTlR_.emplace_back(mytr.tl);
    moduleHR_.emplace_back(mytr.h);
    moduleDzR_.emplace_back(mytr.dz);
    moduleAlphaR_.emplace_back(mytr.alpha);
    moduleCellR_.emplace_back(mytr.cellSize);
  } else {
    moduleLayS_.emplace_back(mytr.lay);
    moduleBlS_.emplace_back(mytr.bl);
    moduleTlS_.emplace_back(mytr.tl);
    moduleHS_.emplace_back(mytr.h);
    moduleDzS_.emplace_back(mytr.dz);
    moduleAlphaS_.emplace_back(mytr.alpha);
    moduleCellS_.emplace_back(mytr.cellSize);
  }
}

HGCalTBParameters::hgtrap HGCalTBParameters::getModule(unsigned int k, bool reco) const {
  HGCalTBParameters::hgtrap mytr;
  if (reco) {
    if (k < moduleLayR_.size()) {
      mytr.lay = moduleLayR_[k];
      mytr.bl = moduleBlR_[k];
      mytr.tl = moduleTlR_[k];
      mytr.h = moduleHR_[k];
      mytr.dz = moduleDzR_[k];
      mytr.alpha = moduleAlphaR_[k];
      mytr.cellSize = moduleCellR_[k];
    } else {
      mytr.lay = -1;
      mytr.bl = mytr.tl = mytr.h = mytr.dz = mytr.alpha = mytr.cellSize = 0;
    }
  } else {
    if (k < moduleLayS_.size()) {
      mytr.lay = moduleLayS_[k];
      mytr.bl = moduleBlS_[k];
      mytr.tl = moduleTlS_[k];
      mytr.h = moduleHS_[k];
      mytr.dz = moduleDzS_[k];
      mytr.alpha = moduleAlphaS_[k];
      mytr.cellSize = moduleCellS_[k];
    } else {
      mytr.lay = -1;
      mytr.bl = mytr.tl = mytr.h = mytr.dz = mytr.alpha = mytr.cellSize = 0;
    }
  }
  return mytr;
}

void HGCalTBParameters::fillTrForm(const HGCalTBParameters::hgtrform& mytr) {
  int zp = (mytr.zp == 1) ? 1 : 0;
  uint32_t indx = ((zp & kMaskZside) << kShiftZside);
  indx |= ((mytr.lay & kMaskLayer) << kShiftLayer);
  indx |= ((mytr.sec & kMaskSector) << kShiftSector);
  indx |= ((mytr.subsec & kMaskSubSec) << kShiftSubSec);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "ZP " << zp << ":" << kMaskZside << ":" << kShiftZside
                                << ((zp & kMaskZside) << kShiftZside) << " Lay " << mytr.lay << ":" << kMaskLayer << ":"
                                << kShiftLayer << ":" << ((mytr.lay & kMaskLayer) << kShiftLayer) << " Sector "
                                << mytr.sec << ":" << kMaskSector << ":" << kShiftSector << ":"
                                << ((mytr.sec & kMaskSector) << kShiftSector) << " SubSec " << mytr.subsec << ":"
                                << kMaskSubSec << ":" << kShiftSubSec << ":"
                                << ((mytr.subsec & kMaskSubSec) << kShiftSubSec) << " Index " << std::hex << indx
                                << std::dec;
#endif
  trformIndex_.emplace_back(indx);
  trformTranX_.emplace_back(mytr.h3v.x());
  trformTranY_.emplace_back(mytr.h3v.y());
  trformTranZ_.emplace_back(mytr.h3v.z());
  trformRotXX_.emplace_back((std::abs(mytr.hr.xx()) > tol) ? mytr.hr.xx() : 0);
  trformRotYX_.emplace_back((std::abs(mytr.hr.yx()) > tol) ? mytr.hr.yx() : 0);
  trformRotZX_.emplace_back((std::abs(mytr.hr.zx()) > tol) ? mytr.hr.zx() : 0);
  trformRotXY_.emplace_back((std::abs(mytr.hr.xy()) > tol) ? mytr.hr.xy() : 0);
  trformRotYY_.emplace_back((std::abs(mytr.hr.yy()) > tol) ? mytr.hr.yy() : 0);
  trformRotZY_.emplace_back((std::abs(mytr.hr.zy()) > tol) ? mytr.hr.zy() : 0);
  trformRotXZ_.emplace_back((std::abs(mytr.hr.xz()) > tol) ? mytr.hr.xz() : 0);
  trformRotYZ_.emplace_back((std::abs(mytr.hr.yz()) > tol) ? mytr.hr.yz() : 0);
  trformRotZZ_.emplace_back((std::abs(mytr.hr.zz()) > tol) ? mytr.hr.zz() : 0);
#ifdef EDM_ML_DEBUG
  unsigned int k = trformIndex_.size() - 1;
  edm::LogVerbatim("HGCalGeom") << "HGCalTBParameters[" << k << "] Index " << std::hex << trformIndex_[k] << std::dec
                                << " (" << mytr.zp << ", " << mytr.lay << ", " << mytr.sec << ", " << mytr.subsec
                                << ") Translation (" << trformTranX_[k] << ", " << trformTranY_[k] << ", "
                                << trformTranZ_[k] << ") Rotation (" << trformRotXX_[k] << ", " << trformRotYX_[k]
                                << ", " << trformRotZX_[k] << ", " << trformRotXY_[k] << ", " << trformRotYY_[k] << ", "
                                << trformRotZY_[k] << ", " << trformRotXZ_[k] << ", " << trformRotYZ_[k] << ", "
                                << trformRotZZ_[k];
#endif
}

HGCalTBParameters::hgtrform HGCalTBParameters::getTrForm(unsigned int k) const {
  HGCalTBParameters::hgtrform mytr;
  if (k < trformIndex_.size()) {
    const auto& id = getID(k);
    mytr.zp = id[0];
    mytr.lay = id[1];
    mytr.sec = id[2];
    mytr.subsec = id[3];
    mytr.h3v = CLHEP::Hep3Vector(trformTranX_[k], trformTranY_[k], trformTranZ_[k]);
    const CLHEP::HepRep3x3 rotation(trformRotXX_[k],
                                    trformRotXY_[k],
                                    trformRotXZ_[k],
                                    trformRotYX_[k],
                                    trformRotYY_[k],
                                    trformRotYZ_[k],
                                    trformRotZX_[k],
                                    trformRotZY_[k],
                                    trformRotZZ_[k]);
    mytr.hr = CLHEP::HepRotation(rotation);
  } else {
    mytr.zp = mytr.lay = mytr.sec = mytr.subsec = 0;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBParameters[" << k << "] Index " << std::hex << trformIndex_[k] << std::dec
                                << " (" << mytr.zp << ", " << mytr.lay << ", " << mytr.sec << ", " << mytr.subsec
                                << ") Translation (" << mytr.h3v.x() << ", " << mytr.h3v.y() << ", " << mytr.h3v.z()
                                << ") Rotation (" << mytr.hr.xx() << ", " << mytr.hr.yx() << ", " << mytr.hr.zx()
                                << ", " << mytr.hr.xy() << ", " << mytr.hr.yy() << ", " << mytr.hr.zy() << ", "
                                << mytr.hr.xz() << ", " << mytr.hr.yz() << ", " << mytr.hr.zz();
#endif
  return mytr;
}

void HGCalTBParameters::addTrForm(const CLHEP::Hep3Vector& h3v) {
  unsigned int k = trformTranX_.size();
  if (k > 0) {
    trformTranX_[k - 1] += h3v.x();
    trformTranY_[k - 1] += h3v.y();
    trformTranZ_[k - 1] += h3v.z();
  }
}

void HGCalTBParameters::scaleTrForm(double scale) {
  unsigned int k = trformTranX_.size();
  if (k > 0) {
    trformTranX_[k - 1] *= scale;
    trformTranY_[k - 1] *= scale;
    trformTranZ_[k - 1] *= scale;
  }
}

std::array<int, 4> HGCalTBParameters::getID(unsigned int k) const {
  int zp = ((trformIndex_[k] >> kShiftZside) & kMaskZside);
  if (zp != 1)
    zp = -1;
  int lay = ((trformIndex_[k] >> kShiftLayer) & kMaskLayer);
  int sec = ((trformIndex_[k] >> kShiftSector) & kMaskSector);
  int subsec = ((trformIndex_[k] >> kShiftSubSec) & kMaskSubSec);
  return std::array<int, 4>{{zp, lay, sec, subsec}};
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTBParameters);
