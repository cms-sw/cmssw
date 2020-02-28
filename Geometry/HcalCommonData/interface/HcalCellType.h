///////////////////////////////////////////////////////////////////////////////
// File: HcalCellType.h
// Description: Hcal readout cell definition given by eta boundary and depth
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalCellType_h
#define HcalCellType_h

#include <vector>
#include <iostream>
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

class HcalCellType {
public:
  struct HcalCell {
    bool ok;
    double eta, deta, phi, dphi, rz, drz;
    bool flagrz;
    HcalCell(bool fl = false,
             double et = 0,
             double det = 0,
             double fi = 0,
             double dfi = 0,
             double rzv = 0,
             double drzv = 0,
             bool frz = true)
        : ok(fl), eta(et), deta(det), phi(fi), dphi(dfi), rz(rzv), drz(drzv), flagrz(frz) {}
  };

  HcalCellType(HcalSubdetector detType,
               int etaBin,
               int zside,
               int depthSegment,
               const HcalCell& cell,
               int readoutDirection = 0,
               double samplingFactor = 0,
               double halfSize = 0);
  HcalCellType(const HcalCellType& right);
  const HcalCellType& operator=(const HcalCellType& right);
  ~HcalCellType();

  /// 1=HB, 2=HE, 3=HO, 4=HF (sub detector type)
  /// as in DataFormats/HcalDetId/interface/HcalSubdetector.h
  HcalSubdetector detType() const { return theDetType; }

  /// which eta ring it belongs to, starting from one
  int etaBin() const { return theEtaBin; }
  int zside() const { return theSide; }
  void setEta(int bin, double etamin, double etamax);

  /// which depth segment it is, starting from 1
  /// absolute within the tower, so HE depth of the
  /// overlap doesn't start at 1.
  int depthSegment() const { return theDepthSegment; }
  void setDepth(int bin, double dmin, double dmax);

  /// the number of these cells in a ring
  int nPhiBins() const { return (int)(thePhis.size()); }
  int nPhiModule() const { return static_cast<int>(20. * CLHEP::deg / thePhiBinWidth); }

  /// phi bin width
  double phiBinWidth() const { return thePhiBinWidth; }
  double phiOffset() const { return thePhiOffset; }
  int unitPhi() const { return theUnitPhi; }
  void setPhi(const std::vector<std::pair<int, double>>& phis,
              const std::vector<int>& iphiMiss,
              double foff,
              double dphi,
              int unit);
  void setPhi(const std::vector<std::pair<int, double>>& phis) { thePhis = phis; }

  /// which cell will actually do the readout for this cell
  /// 1 means move hits in this cell up, and -1 means down
  /// 0 means do nothing
  int actualReadoutDirection() const { return theActualReadoutDirection; }

  /// lower cell edge.  Always positive
  double etaMin() const { return theEtaMin; }

  /// cell edge, always positive & greater than etaMin
  double etaMax() const { return theEtaMax; }

  /// Phi modules and the central phi values
  std::vector<std::pair<int, double>> phis() const { return thePhis; }

  /// z or r position, depending on whether it's barrel or endcap
  double depth() const { return (theDepthMin + theDepthMax) / 2; }
  double depthMin() const { return theDepthMin; }
  double depthMax() const { return theDepthMax; }
  bool depthType() const { return theRzFlag; }
  double halfSize() const { return theHalfSize; }

  /// ratio of real particle energy to deposited energy in the SimHi
  double samplingFactor() const { return theSamplingFactor; }

protected:
  HcalCellType();

private:
  HcalSubdetector theDetType;
  int theEtaBin;
  int theSide;
  int theDepthSegment;
  int theUnitPhi;
  int theActualReadoutDirection;

  bool theRzFlag;

  double theEtaMin;
  double theEtaMax;
  double theDepthMin;
  double theDepthMax;
  double thePhiOffset;
  double thePhiBinWidth;
  double theHalfSize;
  double theSamplingFactor;

  std::vector<std::pair<int, double>> thePhis;
};

std::ostream& operator<<(std::ostream&, const HcalCellType&);
#endif
