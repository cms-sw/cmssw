///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.h
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalNumberingFromDDD_h
#define HcalNumberingFromDDD_h

#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CLHEP/Vector/ThreeVector.h"

#include <vector>
#include <string>

class DDCompactView;    

class HcalNumberingFromDDD {

public:

  HcalNumberingFromDDD(std::string & name, const DDCompactView & cpv);
  ~HcalNumberingFromDDD();
	 
  struct HcalID {
    int subdet, zside, depth, etaR, phi, phis, lay;
    HcalID(int det=0, int zs=0, int d=0, int et=0, int fi=0, int phiskip=0, int ly=-1) :
      subdet(det), zside(zs), depth(d), etaR(et), phi(fi), phis(phiskip), lay(ly) {}
  };

  unsigned int   numberOfCells(HcalSubdetector subdet) const { 
    return hcalConstants->numberOfCells(subdet); }
  std::vector<HcalCellType> HcalCellTypes(HcalSubdetector subdet) const {
    return hcalConstants->HcalCellTypes(subdet, -1, -1); }
  HcalID         unitID(int det, const CLHEP::Hep3Vector& pos, int depth, int lay=-1) const;
  HcalID         unitID(double eta, double phi, int depth=1, int lay=-1) const;
  HcalID         unitID(int det, double etaR, double phi, int depth,
			int lay=-1) const;
  HcalID         unitID(int det, int zside, int depth, int etaR, int phi, 
			int lay=-1) const;
  HcalCellType::HcalCell cell(int det, int zside, int depth, int etaR, 
			      int iphi) const;
  const HcalDDDSimConstants& ddConstants() const {return *hcalConstants;}

private:

  HcalDDDSimConstants *hcalConstants;
};

#endif
