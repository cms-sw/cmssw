///////////////////////////////////////////////////////////////////////////////
// File: HcalCellType.cc
// Description: Individual readout cell description for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iomanip>

HcalCellType::HcalCellType(HcalSubdetector detType, int etaBin, int phiBin, 
			   int depthSegment, HcalCellType::HcalCell cell, 
			   int readoutDirection, double samplingFactor,
			   int numberZ, int nmodule) :
  theDetType(detType), theEtaBin(etaBin), theDepthSegment(depthSegment),
  theNumberOfZ(numberZ), theActualReadoutDirection(readoutDirection), 
  theSamplingFactor(samplingFactor) {

  theEtaMin   = cell.eta - cell.deta;
  theEtaMax   = cell.eta + cell.deta;
  theRzFlag   = cell.flagrz;
  theDepthMin = cell.rz  - cell.drz;
  theDepthMax = cell.rz  + cell.drz;
  int nphi           = (int)(10*deg/cell.dphi);
  theNumberOfPhiBins = nphi*nmodule;
  double phimin      = cell.phi - cell.dphi;
  thePhiOffset       = (phimin - 2*(phiBin-1)*cell.dphi)/deg;
  thePhiBinWidth     = 2*(cell.dphi)/deg;
}

HcalCellType::HcalCellType(const HcalCellType &right) {

  theDetType                = right.theDetType;
  theEtaBin                 = right.theEtaBin;
  theDepthSegment           = right.theDepthSegment;
  theNumberOfPhiBins        = right.theNumberOfPhiBins;
  theNumberOfZ              = right.theNumberOfZ;
  theActualReadoutDirection = right.theActualReadoutDirection;
  theRzFlag                 = right.theRzFlag;
  theEtaMin                 = right.theEtaMin;
  theEtaMax                 = right.theEtaMax;
  thePhiOffset              = right.thePhiOffset;
  thePhiBinWidth            = right.thePhiBinWidth;
  theDepthMin               = right.theDepthMin;
  theDepthMax               = right.theDepthMax;
  theSamplingFactor         = right.theSamplingFactor;
}

HcalCellType::~HcalCellType() {}

std::ostream& operator<<(std::ostream& os, const HcalCellType& cell) {
  os << "Detector " << cell.detType() << " Eta " << cell.etaBin() << " (" 
     << cell.etaMin() << ":" << cell.etaMax() << ") Depth " 
     << cell.depthSegment() << " (" << cell.depthMin() << ":" 
     << cell.depthMax() << "; " << cell.depthType() << ") Phi " 
     << cell.nPhiBins() << " ("	<< cell.phiOffset() << ", "
     << cell.phiBinWidth() << ", " << cell.nPhiModule() << ") Halves " 
     << cell.nHalves() << " Direction " << cell.actualReadoutDirection()
     << " Sampling Factor " << cell.samplingFactor();
  return os;
}
