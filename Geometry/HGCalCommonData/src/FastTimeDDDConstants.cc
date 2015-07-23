#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"

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

FastTimeDDDConstants::FastTimeDDDConstants(const DDCompactView& cpv) {

#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "FastTimeDDDConstants::FastTimeDDDConstants ( const DDCompactView& cpv ) constructor";
#endif
  initialize(cpv);

}

FastTimeDDDConstants::~FastTimeDDDConstants() { 
#ifdef DebugLog
  std::cout << "FastTimeDDDConstants:destructed!!!" << std::endl;
#endif
}

int FastTimeDDDConstants::computeCells() const {

  int copy(0);
#ifdef DebugLog
  int column(0), rowmax(0);
#endif
  double offsetX (0), offsetY(0);
  while (offsetX < rOut) {
#ifdef DebugLog
    column++;
    int row(0);
#endif
    while (offsetY <rOut) {
#ifdef DebugLog
      row++;
#endif
      double limit1 = sqrt((offsetX+cellSize)*(offsetX+cellSize) +
			   (offsetY+cellSize)*(offsetY+cellSize));
      double limit2 = sqrt(offsetX*offsetX+offsetY*offsetY);
      if (limit2 > rIn && limit1 < rOut) copy++;
      offsetY += cellSize;
    }
#ifdef DebugLog
    if (row > rowmax) rowmax = row;
#endif
    offsetY  = 0;
    offsetX += cellSize;
  }
#ifdef DebugLog
  std::cout << rowmax << " rows and " << column << " columns with total of "
	    << copy << " cells in a quadrant " << std::endl;
#endif
  return 4*copy;
}

std::pair<int,int> FastTimeDDDConstants::getXY(int copy) const {

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
    int iy = (iq == 1 || iq == 2) ? (jy + nRows) : (nRows+1-jy);
    return std::pair<int,int>(ix,iy);
  } else {
    return std::pair<int,int>(0,0);
  }
}

std::pair<int,int> FastTimeDDDConstants::getXY(double x, double y) const {

  int jx = floor(fabs(x)/cellSize);
  int jy = floor(fabs(y)/cellSize);
  int iq(0);
  if (x < 0) {
    if (y < 0) iq = 3;
    else       iq = 2;
  } else {
    if (y < 0) iq = 4;
    else       iq = 1;
  }
  int ix = (iq == 1 || iq == 4) ? (jx + nCols) : (nCols+1-jx);
  int iy = (iq == 1 || iq == 2) ? (jy + nRows) : (nRows+1-jy);
  if (isValidXY(ix,iy)) {
    return std::pair<int,int>(ix,iy);
  } else {
    return std::pair<int,int>(0,0);
  }
}

bool FastTimeDDDConstants::isValidXY(int ix, int iy) const {
  int  iq = quadrant(ix,iy);
  if (iq != 0) {
    int kx = (iq == 1 || iq == 4) ? (ix-nCols) : (nCols-1-ix);
    int ky = (iq == 1 || iq == 2) ? (iy-nRows) : (nRows-1-iy);
    bool ok = (ky+1 >= firstY[kx] && ky+1 <= lastY[kx]);
    return ok;
  } else {
    return false;
  }
}

bool FastTimeDDDConstants::isValidCell(int copy) const {
  bool ok = (copy > 0 && copy <= getCells());
  return ok;
}

int FastTimeDDDConstants::quadrant(int ix, int iy) const {
  int iq(0);
  if (ix>nCols && ix<=2*nCols) {
    if (iy>nRows && iy<=2*nRows) iq = 1;
    else if (iy>0 && iy<=nRows)  iq = 4;
  } else if (ix>0 && ix<=nCols) {
    if (iy>nRows && iy<=2*nRows) iq = 2;
    else if (iy>0 && iy<=nRows)  iq = 3;
  }
  return iq;
}

int FastTimeDDDConstants::quadrant(int copy) const {
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

void FastTimeDDDConstants::initialize(const DDCompactView& cpv) {

  std::string attribute = "Volume"; 
  std::string value     = "SFBX";
  DDValue val(attribute, value, 0.0);
  
  DDSpecificsFilter filter;
  filter.setCriteria(val, DDCompOp::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool ok = fv.firstChild();

  if (ok) {
    loadSpecPars(fv);
  } else {
    edm::LogError("HGCalGeom") << "FastTimeDDDConstants: cannot get filtered"
			       << " view for " << attribute 
			       << " not matching " << value;
    throw cms::Exception("DDException") << "FastTimeDDDConstants: cannot match " << attribute << " to " << value;
  }
}

void FastTimeDDDConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  // First and Last Row number in each column
  firstY = dbl_to_int(getDDDArray("firstRow",sv));
  lastY  = dbl_to_int(getDDDArray("lastRow", sv));
  if (firstY.size() != lastY.size()) {
    edm::LogError("HGCalGeom") << "FastTimeDDDConstants: unequal numbers "
			       << firstY.size() << ":" << lastY.size()
			       << " elements for first and last rows";
    throw cms::Exception("DDException") << "FastTimeDDDConstants: wrong array sizes for first/last Row";
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
  std::cout << "FastTimeDDDConstants: nCells = " << nCells << ", nRow = " 
	    << 2*nRows << ", nColumns = " << 2*nCols << std::endl;
  for (int k=0; k<nCols; ++k) 
    std::cout << "Column[" << k << "] Cells = " << firstCell[k] << ":" 
	      << lastCell[k] << ", Rows = " << firstY[k] << ":" << lastY[k] 
	      << std::endl;
#endif

  std::vector<double> gpar = getDDDArray("geomPars",sv);
  if (gpar.size() < 3) {
    edm::LogError("HGCalGeom") << "FastTimeDDDConstants: too few "
			       << gpar.size() << " elements for gpar";
    throw cms::Exception("DDException") << "FastTimeDDDConstants: wrong array sizes for gpar";
  }
  rIn      = gpar[0];
  rOut     = gpar[1];
  cellSize = gpar[2];
#ifdef DebugLog
  std::cout << "FastTimeDDDConstants: cellsize " << cellSize << " in region "
	    << rIn << ":" << rOut << std::endl;
#endif

  gpar     = getDDDArray("geomType",sv);
  cellType = int(gpar[0]);
#ifdef DebugLog
  std::cout << "FastTimeDDDConstants: cell type " << cellType << std::endl;
#endif
}

std::vector<double> FastTimeDDDConstants::getDDDArray(const std::string & str, 
						      const DDsvalues_type & sv) const {

#ifdef DebugLog
  std::cout << "FastTimeDDDConstants:getDDDArray called for " << str << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << "FastTimeDDDConstants: " << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval > 0) return fvec;
  }
  edm::LogError("HGCalGeom") << "FastTimeDDDConstants: cannot get array " 
			     << str;
  throw cms::Exception("DDException") << "FastTimeDDDConstants: cannot get array " << str;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(FastTimeDDDConstants);
