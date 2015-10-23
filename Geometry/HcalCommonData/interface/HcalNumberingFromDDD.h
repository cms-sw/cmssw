///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingFromDDD.h
// Description: Usage of DDD to get to numbering scheme for hadron calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalNumberingFromDDD_h
#define HcalNumberingFromDDD_h

#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CLHEP/Vector/ThreeVector.h"

#include <vector>
#include <string>

class HcalNumberingFromDDD {

public:

  HcalNumberingFromDDD(HcalDDDSimConstants * hcons);
  ~HcalNumberingFromDDD();
	 
  struct HcalID {
    int subdet, zside, depth, etaR, phi, phis, lay;
    HcalID(int det=0, int zs=0, int d=0, int et=0, int fi=0, int phiskip=0, int ly=-1) :
      subdet(det), zside(zs), depth(d), etaR(et), phi(fi), phis(phiskip), lay(ly) {}
  };

  HcalID         unitID(int det, const CLHEP::Hep3Vector& pos, int depth, int lay=-1) const;
  HcalID         unitID(double eta, double phi, int depth=1, int lay=-1) const;
  HcalID         unitID(int det, double etaR, double phi, int depth,
			int lay=-1) const;
  HcalID         unitID(int det, int zside, int depth, int etaR, int phi, 
			int lay=-1) const;

private:

  HcalDDDSimConstants *hcalConstants;
};

#endif
