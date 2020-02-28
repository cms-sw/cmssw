#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include <cmath>
PreshowerLayer2Properties::PreshowerLayer2Properties(const edm::ParameterSet& fastDet) : PreshowerProperties() {
  // Preshower : mumber of Mips / GeV
  mips = fastDet.getParameter<double>("PreshowerLayer2_mipsPerGeV");
  thick = fastDet.getParameter<double>("PreshowerLayer2_thickness");
  pseeradLenIncm_ = fastDet.getUntrackedParameter<double>("PreshowerEEGapRadLenInCm", 63.);
  pseeInteractionLength_ = fastDet.getUntrackedParameter<double>("PreshowerEEGapIntLenInCm", 111.);
}

double PreshowerLayer2Properties::thickness(const double eta) const {
  // eta is the pseudorapidity
  double e = exp(-eta);
  double e2 = e * e;
  // 1 / cos theta
  double cinv = (1. + e2) / (1. - e2);
  //    double c  = (1.-e2)/(1.+e2);
  //    double s  = 2.*e/(1.+e2);
  //    double t  = 2.*e/(1.-e2);
  double feta = fabs(eta);

  if (1.637 < feta && feta < 2.625) {
    return thick * fabs(cinv);
  } else {
    return 0;
  }
}
