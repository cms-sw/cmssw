#include "Geometry/HGCalCommonData/interface/FastTimingDDDConstants.h"

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

FastTimingDDDConstants::FastTimingDDDConstants() : tobeInitialized(true), 
						   nCells(0) {

#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "FastTimingDDDConstants::FastTimingDDDConstants constructor";
#endif

}

FastTimingDDDConstants::FastTimingDDDConstants(const DDCompactView& cpv) : tobeInitialized(true) {

#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "FastTimingDDDConstants::FastTimingDDDConstants ( const DDCompactView& cpv ) constructor";
#endif
  initialize(cpv);

}

FastTimingDDDConstants::~FastTimingDDDConstants() { 
#ifdef DebugLog
  std::cout << "FastTimingDDDConstants:destructed!!!" << std::endl;
#endif
}

std::pair<int,int> FastTimingDDDConstants::getXY(int copy) const {

  int iq = quadrant(copy);
  if (iq != 0) {
    int ism = copy - (iq-1)*nCells;
    int jx(0), jy(0);
    for (unsigned int k=0; k<firstY.size(); ++k) {
      if (ism >= firstCell[k] && ism <= lastCell[k]) {
	jx = k + 1;
	jy = ism - firstCell[k] + firstY[k];
	break;
      }
    }
    int ix = (iq == 1 || iq == 4) ? (jx + nCols) : (nCols+1-jx);
    int iy = (iq == 1 || iq == 2) ? (jy + nCols) : (nCols+1-jy);
    return std::pair<int,int>(ix,iy);
  } else {
    return std::pair<int,int>(0,0);
  }
}

std::pair<int,int> FastTimingDDDConstants::getXY(double x, double y) const {

  int ix = floor(fabs(x)/cellSize);
  int iy = floor(fabs(y)/cellSize);
  if (x>0) ix += nCols;
  if (y>0) iy += nRows;
  if (isValidXY(ix,iy)) {
    return std::pair<int,int>(ix,iy);
  } else {
    return std::pair<int,int>(0,0);
  }
}

void FastTimingDDDConstants::initialize(const DDCompactView& cpv) {

  if (tobeInitialized) {
    tobeInitialized = false;

    std::string attribute = "OnlyForFastTimingNumbering"; 
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
      edm::LogError("HGCalGeom") << "FastTimingDDDConstants: cannot get filtered"
				 << " view for " << attribute 
				 << " not matching " << value;
      throw cms::Exception("DDException") << "FastTimingDDDConstants: cannot match " << attribute << " to " << value;
    }
  }
}

bool FastTimingDDDConstants::isValidXY(int ix, int iy) const {
  int  iq = quadrant(ix,iy);
  if (iq != 0) {
    int kx = (iq == 1 || iq == 4) ? (ix-nRows) : (nRows-1-ix);
    int ky = (iq == 1 || iq == 2) ? (iy-nCols) : (nCols-1-iy);
    bool ok = (ky+1 >= firstY[kx] && ky+1 <= lastY[kx]);
    return ok;
  } else {
    return false;
  }
}

bool FastTimingDDDConstants::isValidCell(int copy) const {
  bool ok = (copy > 0 && copy <= getCells());
  return ok;
}

int FastTimingDDDConstants::quadrant(int ix, int iy) const {
  int iq(0);
  if (ix>nRows && ix<=2*nRows) {
    if (iy>nCols && iy<=2*nCols) iq = 1;
    else if (iy>0 && iy<=nCols)  iq = 4;
  } else if (ix>0 && ix<=nRows) {
    if (iy>nCols && iy<=2*nCols) iq = 2;
    else if (iy>0 && iy<=nCols)  iq = 3;
  }
  return iq;
}

int FastTimingDDDConstants::quadrant(int copy) const {
  int iq(0);
  if (copy > 4*nCells) {
  } else if (copy > 3*nCells) {
    iq = 4;
  } else if (copy > 2*nCells) {
    iq = 3;
  } else if (copy > nCells) {
    iq = 2;
  } else if (copy > 0) {
    iq = 1;
  }
  return iq;
}

void FastTimingDDDConstants::checkInitialized() const {
  if (tobeInitialized) {
    edm::LogError("HGCalGeom") << "FastTimingDDDConstants : to be initialized correctly";
    throw cms::Exception("DDException") << "FastTimingDDDConstants: to be initialized";
  }
} 

void FastTimingDDDConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // First and Last Row number in each column
  firstY = dbl_to_int(getDDDArray("firstRow",sv));
  lastY  = dbl_to_int(getDDDArray("lastRow", sv));
  if (firstY.size() != lastY.size()) {
    edm::LogError("HGCalGeom") << "FastTimingDDDConstants: unequal numbers "
			       << firstY.size() << ":" << lastY.size()
			       << " elements for first and last rows";
    throw cms::Exception("DDException") << "FastTimingDDDConstants: wrong array sizes for first/last Row";
  }

  nCells = 0;
  nCols  = (int)(firstY.size());
  nRows  = 0;
  for (int k=0; k<nCols; ++k) {
    firstCell.push_back(nCells+1);
    nCells += (lastY[k]-firstY[k]+1);
    lastCell.push_back(nCells);
    if (lastY[k] > nRows) nRows = lastY[k];
  }

#ifdef DebugLog
  std::cout << "FastTimingDDDConstants: nCells = " << nCells << ", nRow = " 
	    << 2*nRows << ", nColumns = " << 2*nColS << std::endl;
  for (int k=0; k<nCols; ++k) 
    std::cout << "Column[" << k << "] Cells = " << firstCell[k] << ":" 
	      << lastCell[k] << ", Rows = " << firstY[k] << ":" << lastY[k] 
	      << std::endl;
#endif

  std::vector<double> gpar = getDDDArray("gparCell",sv);
  if (gpar.size() < 3) {
    edm::LogError("HGCalGeom") << "FastTimingDDDConstants: too few "
			       << gpar.size() << " elements for gpar";
    throw cms::Exception("DDException") << "FastTimingDDDConstants: wrong array sizes for gpar";
  }
  rIn      = gpar[0];
  rOut     = gpar[1];
  cellSize = gpar[2];
#ifdef DebugLog
  std::cout << "FastTimingDDDConstants: cellsize " << cellSize << " in region "
	    << rIn << ":" << rOut << std::endl;
#endif
}

std::vector<double> FastTimingDDDConstants::getDDDArray(const std::string & str, 
							const DDsvalues_type & sv) const {

#ifdef DebugLog
  std::cout << "FastTimingDDDConstants:getDDDArray called for " << str << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "FastTimingDDDConstants: " << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval > 0) return fvec;
  }
  edm::LogError("HGCalGeom") << "FastTimingDDDConstants: cannot get array " 
			     << str;
  throw cms::Exception("DDException") << "FastTimingDDDConstants: cannot get array " << str;
}
