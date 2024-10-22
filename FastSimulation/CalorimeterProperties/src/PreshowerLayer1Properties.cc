#include "FWCore/ParameterSet/interface/ParameterSet.h"
//This class header
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include <cmath>
PreshowerLayer1Properties::PreshowerLayer1Properties(const edm::ParameterSet& fastDet) : PreshowerProperties() {
  // Preshower : mumber of Mips / GeV
  mips = fastDet.getParameter<double>("PreshowerLayer1_mipsPerGeV");
  thick = fastDet.getParameter<double>("PreshowerLayer1_thickness");
}

double PreshowerLayer1Properties::thickness(double eta) const {
  // eta is the pseudorapidity
  double e = exp(-eta);
  double e2 = e * e;
  // 1 / cos theta
  double cinv = (1. + e2) / (1. - e2);
  //    double c  = (1.-e2)/(1.+e2);
  //    double s  = 2.*e/(1.+e2);
  //    double t  = 2.*e/(1.-e2);
  double feta = fabs(eta);

  if (1.623 < feta && feta < 2.611) {
    return thick * fabs(cinv);
  } else {
    return 0;
  }
}
