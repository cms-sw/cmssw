#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

ShashlikDDDConstants::ShashlikDDDConstants() : tobeInitialized(true), nSM(0),
					       nColS(0) {

#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "ShashlikDDDConstants::ShashlikDDDConstants constructor";
#endif

}

ShashlikDDDConstants::ShashlikDDDConstants(const DDCompactView& cpv) : tobeInitialized(true) {

#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "ShashlikDDDConstants::ShashlikDDDConstants ( const DDCompactView& cpv ) constructor";
#endif

  initialize(cpv);

}


ShashlikDDDConstants::~ShashlikDDDConstants() { 
#ifdef DebugLog
  std::cout << "destructed!!!" << std::endl;
#endif
}

std::pair<int,int> ShashlikDDDConstants::getSMM(int ix, int iy) const {

  int iq = quadrant(ix,iy);
  if (iq != 0) {
    int jx = (ix-1)/nMods;
    int jy = (iy-1)/nMods;
    int kx = (iq == 1 || iq == 4) ? (jx-nColS) : (nColS-1-jx);
    int ky = (iq == 1 || iq == 2) ? (jy-nColS) : (nColS-1-jy);
    int sm = (ky+1 >= firstY[kx] && ky+1 <= lastY[kx]) ? firstSM[kx]+ky-firstY[kx]+1+(iq-1)*nSM : 0;
    int mod= (sm > 0) ? (((ix-1)%nMods) + ((iy-1)%nMods)*nMods + 1) : 0;
    return std::pair<int,int>(sm,mod);
  } else {
    return std::pair<int,int>(0,0);
  }
}

std::pair<int,int> ShashlikDDDConstants::getXY(int sm, int mod) const {

  int iq = quadrant(sm);
  if (iq != 0) {
    int ism = sm - (iq-1)*nSM;
    int jx(0), jy(0);
    for (unsigned int k=0; k<firstY.size(); ++k) {
      if (ism >= firstSM[k] && ism <= lastSM[k]) {
	jx = k + 1;
	jy = ism - firstSM[k] + firstY[k];
	break;
      }
    }
    int kx = (iq == 1 || iq == 4) ? (jx + nColS) : (nColS+1-jx);
    int ky = (iq == 1 || iq == 2) ? (jy + nColS) : (nColS+1-jy);
    int ll = (mod-1)/nMods;
    int ix = (kx-1)*nMods+(mod-ll*nMods);
    int iy = (ky-1)*nMods+(ll+1);
    return std::pair<int,int>(ix,iy);
  } else {
    return std::pair<int,int>(0,0);
  }
}

void ShashlikDDDConstants::initialize(const DDCompactView& cpv) {

  if (tobeInitialized) {
    tobeInitialized = false;

    std::string attribute = "OnlyForShashlikNumbering"; 
    std::string value     = "any";
    DDValue val(attribute, value, 0.0);
  
    DDSpecificsFilter filter;
    filter.setCriteria(val, DDSpecificsFilter::not_equals,
                       DDSpecificsFilter::AND, true, true);
    DDFilteredView fv(cpv);
    fv.addFilter(filter);
    bool ok = fv.firstChild();

    if (ok) {
      loadSpecPars(fv);

    } else {
      edm::LogError("HGCalGeom") << "ShashlikDDDConstants: cannot get filtered"
				 << " view for " << attribute 
				 << " not matching " << value;
      throw cms::Exception("DDException") << "ShashlikDDDConstants: cannot match " << attribute << " to " << value;
    }
  }
}

bool ShashlikDDDConstants::isValidXY(int ix, int iy) const {
  int  iq = quadrant(ix,iy);
  if (iq != 0) {
    int jx = (ix-1)/nMods;
    int jy = (iy-1)/nMods;
    int kx = (iq == 1 || iq == 4) ? (jx-nColS) : (nColS-1-jx);
    int ky = (iq == 1 || iq == 2) ? (jy-nColS) : (nColS-1-jy);
    bool ok = (ky+1 >= firstY[kx] && ky+1 <= lastY[kx]);
    return ok;
  } else {
    return false;
  }
}

bool ShashlikDDDConstants::isValidSMM(int sm, int mod) const {
  bool ok = (sm > 0 && sm <= getSuperModules() && mod > 0 && mod <= getModules());
  return ok;
}

int ShashlikDDDConstants::quadrant(int ix, int iy) const {
  int iq(0);
  if (ix>nRow && ix<=2*nRow) {
    if (iy>nRow && iy<=2*nRow) iq = 1;
    else if (iy>0 && iy<=nRow) iq = 4;
  } else if (ix>0 && ix<=nRow) {
    if (iy>nRow && iy<=2*nRow) iq = 2;
    else if (iy>0 && iy<=nRow) iq = 3;
  }
  return iq;
}

int ShashlikDDDConstants::quadrant(int sm) const {
  int iq(0);
  if (sm > 4*nSM) {
  } else if (sm > 3*nSM) {
    iq = 4;
  } else if (sm > 2*nSM) {
    iq = 3;
  } else if (sm > nSM) {
    iq = 2;
  } else if (sm > 0) {
    iq = 1;
  }
  return iq;
}

void ShashlikDDDConstants::checkInitialized() const {
  if (tobeInitialized) {
    edm::LogError("HGCalGeom") << "ShashlikDDDConstants : to be initialized correctly";
    throw cms::Exception("DDException") << "ShashlikDDDConstants: to be initialized";
  }
} 

void ShashlikDDDConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // First and Last Row number in each column
  firstY = dbl_to_int(getDDDArray("firstRow",sv));
  lastY  = dbl_to_int(getDDDArray("lastRow", sv));
  if (firstY.size() != lastY.size()) {
    edm::LogError("HGCalGeom") << "ShashlikDDDConstants: unequal # of columns "
			       << firstY.size() << ":" << lastY.size()
			       << " for first and last rows";
    throw cms::Exception("DDException") << "ShashlikDDDConstants: wrong array sizes for first/last Row";
  }

  nSM   = 0;
  nColS = (int)(firstY.size());
  nRow  = 0;
  for (unsigned int k=0; k<firstY.size(); ++k) {
    firstSM.push_back(nSM+1);
    nSM += (lastY[k]-firstY[k]+1);
    lastSM.push_back(nSM);
    if (lastY[k] > nRow) nRow = lastY[k];
  }

#ifdef DebugLog
  std::cout << "ShashlikDDDConstants: nSM = " << nSM << ", nModule = " 
	    << nMods << ", nRow = " << 2*nRow << ", nColumns = " 
	    << 2*nColS << std::endl;
  for (unsigned int k=0; k<firstY.size(); ++k) 
    std::cout << "Column[" << k << "] SM = " << firstSM[k] << ":" << lastSM[k]
	      << ", Rows = " << firstY[k] << ":" << lastY[k] << std::endl;
#endif
  nRow  = nColS*nMods;
}

std::vector<double> ShashlikDDDConstants::getDDDArray(const std::string & str, 
                                                     const DDsvalues_type & sv) const {

#ifdef DebugLog
  std::cout << "ShashlikDDDConstants:getDDDArray called for " << str << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "ShashlikDDDConstants: " << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval > 0) return fvec;
  }
  edm::LogError("HGCalGeom") << "ShashlikDDDConstants: cannot get array " 
			     << str;
  throw cms::Exception("DDException") << "ShashlikDDDConstants: cannot get array " << str;
}
