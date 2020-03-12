///////////////////////////////////////////////////////////////////////////////
// File: HcalCellType.cc
// Description: Individual readout cell description for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <algorithm>
#include <iomanip>

HcalCellType::HcalCellType(HcalSubdetector detType,
                           int etaBin,
                           int zside,
                           int depthSegment,
                           const HcalCellType::HcalCell& cell,
                           int readoutDirection,
                           double samplingFactor,
                           double halfSize)
    : theDetType(detType),
      theEtaBin(etaBin),
      theSide(zside),
      theDepthSegment(depthSegment),
      theActualReadoutDirection(readoutDirection),
      theSamplingFactor(samplingFactor) {
  theEtaMin = cell.eta - cell.deta;
  theEtaMax = cell.eta + cell.deta;
  theRzFlag = cell.flagrz;
  theDepthMin = (cell.rz - cell.drz) / CLHEP::cm;
  theDepthMax = (cell.rz + cell.drz) / CLHEP::cm;
  thePhiBinWidth = 2 * (cell.dphi) / CLHEP::deg;
  thePhiOffset = 0;
  theUnitPhi = 1;
  theHalfSize = halfSize / CLHEP::cm;
}

HcalCellType::HcalCellType(const HcalCellType& right) {
  theDetType = right.theDetType;
  theEtaBin = right.theEtaBin;
  theSide = right.theSide;
  theUnitPhi = right.theUnitPhi;
  theDepthSegment = right.theDepthSegment;
  theActualReadoutDirection = right.theActualReadoutDirection;
  theRzFlag = right.theRzFlag;
  theEtaMin = right.theEtaMin;
  theEtaMax = right.theEtaMax;
  theDepthMin = right.theDepthMin;
  theDepthMax = right.theDepthMax;
  thePhiBinWidth = right.thePhiBinWidth;
  thePhiOffset = right.thePhiOffset;
  theHalfSize = right.theHalfSize;
  theSamplingFactor = right.theSamplingFactor;
  thePhis = right.thePhis;
}

const HcalCellType& HcalCellType::operator=(const HcalCellType& right) {
  theDetType = right.theDetType;
  theEtaBin = right.theEtaBin;
  theSide = right.theSide;
  theUnitPhi = right.theUnitPhi;
  theDepthSegment = right.theDepthSegment;
  theActualReadoutDirection = right.theActualReadoutDirection;
  theRzFlag = right.theRzFlag;
  theEtaMin = right.theEtaMin;
  theEtaMax = right.theEtaMax;
  theDepthMin = right.theDepthMin;
  theDepthMax = right.theDepthMax;
  thePhiBinWidth = right.thePhiBinWidth;
  thePhiOffset = right.thePhiOffset;
  theHalfSize = right.theHalfSize;
  theSamplingFactor = right.theSamplingFactor;
  thePhis = right.thePhis;

  return *this;
}

HcalCellType::~HcalCellType() {}

void HcalCellType::setEta(int bin, double etamin, double etamax) {
  theEtaBin = bin;
  theEtaMin = etamin;
  theEtaMax = etamax;
}

void HcalCellType::setDepth(int bin, double dmin, double dmax) {
  theDepthSegment = bin;
  theDepthMin = dmin;
  theDepthMax = dmax;
}

void HcalCellType::setPhi(const std::vector<std::pair<int, double> >& phis,
                          const std::vector<int>& iphiMiss,
                          double foff,
                          double dphi,
                          int unit) {
  thePhiBinWidth = dphi;
  thePhiOffset = foff;
  theUnitPhi = unit;
  thePhis.clear();
  for (const auto& phi : phis) {
    if (std::find(iphiMiss.begin(), iphiMiss.end(), phi.first) == iphiMiss.end()) {
      thePhis.emplace_back(phi);
    }
  }
}

std::ostream& operator<<(std::ostream& os, const HcalCellType& cell) {
  os << "Detector " << cell.detType() << " Eta " << cell.etaBin() << " (" << cell.etaMin() << ":" << cell.etaMax()
     << ") Zside " << cell.zside() << " Depth " << cell.depthSegment() << " (" << cell.depthMin() << ":"
     << cell.depthMax() << "; " << cell.depthType() << ") Phi " << cell.nPhiBins() << " ("
     << cell.phiOffset() / CLHEP::deg << ", " << cell.phiBinWidth() / CLHEP::deg << ", " << cell.unitPhi() << ", "
     << cell.nPhiModule() << ")  Direction " << cell.actualReadoutDirection() << " Half size " << cell.halfSize()
     << " Sampling Factor " << cell.samplingFactor();
  return os;
}
