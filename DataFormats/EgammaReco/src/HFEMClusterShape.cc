#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"

reco::HFEMClusterShape::HFEMClusterShape(double energy,
					 double eLong1x1,
					 double eShort1x1,
					 double eLong3x3,double eShort3x3,
					 double eLong5x5,
					 double eShort5x5,double eLongCore,
					 double CellEta,double CellPhi):
  energy_(energy),
  eLong1x1_(eLong1x1),
  eShort1x1_(eShort1x1),
  eLong3x3_(eLong3x3), 
  eShort3x3_(eShort3x3),  
  eLong5x5_(eLong5x5), 
  eShort5x5_(eShort5x5), 
  eLongCore_(eLongCore),
  CellEta_(CellEta), 
  CellPhi_(CellPhi)
{
}

// double HFEMCluster::et() const {
//   return energy_/cosh(eta_);
//}
double reco::HFEMClusterShape::e1x1() const {
  return eLong1x1_+eShort1x1_;
}
double reco::HFEMClusterShape::e3x3() const {
  return eLong3x3_+eShort3x3_;
}
double reco::HFEMClusterShape::e5x5() const {
  return eLong5x5_+eShort5x5_;
}
