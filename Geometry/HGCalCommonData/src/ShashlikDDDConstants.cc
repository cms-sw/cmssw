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

ShashlikDDDConstants::ShashlikDDDConstants(const DDCompactView& cpv) {

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

std::pair<int,int> ShashlikDDDConstants::getSMM(int ix, int iy, bool testOnly) const {

  int iq = quadrant(ix,iy, testOnly);
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

bool ShashlikDDDConstants::isValidXY(int ix, int iy) const {
  int  iq = quadrant(ix,iy, true);
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

int ShashlikDDDConstants::quadrant(int ix, int iy, bool testOnly) const {
  int iq(0);
  ix = (ix-1) / nMods + 1;
  iy = (iy-1) / nMods + 1;
  if (ix>nRow && ix<=2*nRow) {
    if (iy>nRow && iy<=2*nRow) iq = 1;
    else if (iy>0 && iy<=nRow) iq = 4;
  } else if (ix>0 && ix<=nRow) {
    if (iy>nRow && iy<=2*nRow) iq = 2;
    else if (iy>0 && iy<=nRow) iq = 3;
  }
  assert (iq != 0 || testOnly);
  return iq;
}

int ShashlikDDDConstants::quadrant(int sm, bool testOnly) const {
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
  else {
    assert (testOnly);
  }
  return iq;
}

void ShashlikDDDConstants::initialize(const DDCompactView& cpv) {

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

void ShashlikDDDConstants::loadSpecPars(const std::vector<int>& firstY,
					const std::vector<int>& lastY) {
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
  //std::cout << " ShashlikDDDConstants::loadSpecPars-> SM/Rows/Cols: " << nSM << '/' << nRow << '/' << nColS << std::endl;
}
 

void ShashlikDDDConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // First and Last Row number in each column
  firstY = dbl_to_int(getDDDArray("firstRow",sv));
  lastY  = dbl_to_int(getDDDArray("lastRow", sv));
  loadSpecPars (firstY, lastY); 
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
